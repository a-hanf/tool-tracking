"""Definition of tensor train layer(s)."""
from itertools import count
from typing import Dict
from typing import List

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LSTMCell
from tensorflow.python.ops import array_ops

from tensor_train import TensorTrain


class TTDense(Layer):
    """A tensor train dense layer."""

    _counter = count(0)

    def __init__(
        self,
        tt: TensorTrain,
        activation: str = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_normal",
        bias_initializer: float = 1e-2,
        **kwargs: Dict
    ):
        """
        Create a dense layer in tensor train format.

        Parameters
        ----------
        tt : TensorTrain
            A tensor train representation of the weight matrix.
        activation : str
            A standard activation function, e.g. 'relu'.
        use_bias : bool
            Whether to add a bias term to the computation.
        kernel_initializer : str
            A standard initializer for the weight matrix, e.g. 'glorot_normal'.
        bias_initializer : float
            TODO: A constant initializer for the bias.
        kwargs : dict
            Additional arguments accepted by tensorflow.keras.layers.Layer.
        """

        self.counter = next(self._counter)
        name = "tt_dense_{}".format(self.counter)

        self.tt = tt
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.output_dim = self.tt.matrix_shape[1]
        # TODO: maybe move to build?
        self._w = self.init_kernel()

        self.compression_ratio = (
            self.tt.matrix_shape[0]
            * self.tt.matrix_shape[1]
            / self.tt.num_params
        )

        self.b = None
        if self.use_bias:
            # TODO: extend this to a more general initializer
            self.b = tf.Variable(
                self.bias_initializer * tf.ones((self.output_dim,))
            )

        super(TTDense, self).__init__(name=name, **kwargs)

    def init_kernel(self) -> List[tf.Tensor]:
        """
        Initialize cores and return trainable tensors.

        Returns
        -------
        list
            A list of initialized cores
        """
        initializer = tf.keras.initializers.get(self.kernel_initializer)

        def variable_initializer(shape):
            return tf.Variable(
                initializer(shape), dtype="float32", trainable=True
            )

        return self.tt.init_cores(variable_initializer)

    def call(self, inputs: tf.Tensor, **kwargs: Dict) -> tf.Tensor:
        """
        Compute a forward pass of the given inputs.
        """
        res = self.tt.matmul(inputs)
        if self.use_bias:
            res += self.b
        if self.activation is not None:
            res = Activation(self.activation)(res)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class TTLSTMCell(LSTMCell):
    """
    A tensor train LSTM cell.
    Adapted from https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/layers/recurrent.py
    """

    _counter = count(0)

    def __init__(
        self,
        tt: TensorTrain,
        activation: str = "tanh",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_normal",
        bias_initializer: float = 1e-2,
        **kwargs: Dict
    ):
        """
        Create a dense layer in tensor train format.

        Parameters
        ----------
        tt : TensorTrain
            A tensor train representation of the weight matrix.
        activation : str
            A standard activation function, e.g. 'relu'.
        use_bias : bool
            Whether to add a bias term to the computation.
        kernel_initializer : str
            A standard initializer for the weight matrix, e.g. 'glorot_normal'.
        bias_initializer : float
            TODO: A constant initializer for the bias.
        kwargs : dict
            Additional arguments accepted by tensorflow.keras.layers.Layer.
        """
        self.tt = tt
        self.units = self.tt.matrix_shape[1] // 4
        self.counter = next(self._counter)
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self._w = self.init_kernel()

        self.compression_ratio = (
            self.tt.matrix_shape[0]
            * self.tt.matrix_shape[1]
            / self.tt.num_params
        )

        super(TTLSTMCell, self).__init__(
            self.units, name="tt_lstm_cell_{}".format(self.counter), **kwargs
        )

    def init_kernel(self) -> List[tf.Tensor]:
        """
        Initialize cores and return trainable tensors.

        Returns
        -------
        list
            A list of initialized cores
        """
        initializer = tf.keras.initializers.get(self.kernel_initializer)

        def variable_initializer(shape):
            return tf.Variable(
                initializer(shape=shape), dtype="float32", trainable=True
            )

        return self.tt.init_cores(variable_initializer)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        self.built = True

    def call(self, inputs, states, training=None):
        """
        Compute a forward pass of the given inputs.
        """
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        z = self.tt.matmul(inputs)
        z += K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)

        z = array_ops.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)
        if self.activation is not None:
            c = Activation(self.activation)(c)
        h = o * c
        return h, [h, c]
