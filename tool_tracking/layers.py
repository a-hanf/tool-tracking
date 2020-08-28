"""Definition of tensor train layer(s)."""
from itertools import count
from typing import Dict
from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Layer

from tool_tracking.tensor_train import TensorTrain


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
