"""Definition of the tensor train format."""
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
import tensorflow as tf

# import opt_einsum as oe

# from tool_tracking import utils
from tool_tracking.config import logger


class TensorTrain:
    """A tensor train format."""

    def __init__(
        self, input_shape: List[int], output_shape: List[int], ranks: List[int]
    ):
        """
        Create an empty tensor train object.

        Parameters
        ----------
        input_shape : list
            A list of integers [m_1, m_2, ..., m_d]
        output_shape : list
            A list of integers [n_1, n_2, ..., n_d]
        ranks : list
            A list of integers [r_0, r_1, ..., r_d]
        """

        if len(input_shape) != len(output_shape):
            raise ValueError(
                "Invalid input and output shapes! "
                "The input shape must have the same number "
                "of values as the output shape."
            )
        if len(input_shape) + 1 != len(ranks):
            raise ValueError(
                "Invalid ranks! "
                "The ranks must have exactly one more value "
                "than input or output shape."
            )
        if ranks[0] != ranks[-1] or ranks[0] != 1:
            raise ValueError(
                "Invalid ranks! The first and the last rank must equal 1."
            )

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.ranks = ranks

        self.core_count = len(self.input_shape)

        self.core_shapes = [
            [
                self.input_shape[k],
                self.ranks[k],
                self.ranks[k + 1],
                self.output_shape[k],
            ]
            for k in range(self.core_count)
        ]

        self.matrix_shape = [
            np.prod(self.input_shape),
            np.prod(self.output_shape),
        ]

        self.num_params = sum(
            [np.prod(core_shape) for core_shape in self.core_shapes]
        )

        self.cores = None

    @property
    def is_initialized(self) -> bool:
        """
        Check if tensor train cores are initialized.

        Returns
        -------
        bool
            True if cores initialized.
        """
        return self.cores is not None

    def init_cores(self, initializer: Callable) -> List:
        """
            Initialize cores based on the tensor train core shapes.

            Parameters
            ----------
            initializer : callable
                A callable function that accepts at least a shape (tuple or list)
                as its first argument.

            Returns
            -------
            list
                list of initialised cores, same as self.cores.
            """
        if self.is_initialized:
            logger.warning(
                "Cores have already been initialized, "
                "overwriting existing cores!"
            )
        self.cores = [
            initializer(core_shape) for core_shape in self.core_shapes
        ]
        return self.cores

    def matmul(self, matrix: tf.Tensor) -> tf.Tensor:
        """
        Perform matrix multiplication of the tensor train with a matrix.
        The dimensions of the matrix (K x M) must match with the dimensions
        of the tensor train, i.e. M = m_1*m_2*...*m_d.

        Parameters
        ----------
        matrix : array_like
            A two dimensional array of shape K x M.

        Returns
        -------
        tf.Tensor
            The resulting K x N matrix from the multiplication.
        """
        if not self.is_initialized:
            raise ValueError(
                "Cores have not been initialized, "
                "cannot perform multiplication!"
            )

        if not matrix.shape[1] == self.matrix_shape[0]:
            raise ValueError(
                "Matrix shape of %s does not match "
                "the tensor train input shape of %s!"
                % (matrix.shape, self.matrix_shape[0])
            )

        res = matrix
        for core_idx in reversed(range(len(self.cores))):
            curr_core = self.cores[core_idx]
            res = tf.reshape(res, [-1, curr_core.shape[0], curr_core.shape[2]])
            res = tf.einsum("mijn,xmj->nxi", curr_core, res)

        res = tf.reshape(res, (self.matrix_shape[1], -1))
        res = tf.transpose(res)
        return res

    def to_full_matrix(self):
        """
        Convert a tensor train format to its corresponding explicit matrix.

        Returns
        -------
        tf.Tensor
            A matrix (m_1 x m_2 x ... x m_d) x (n_1 x n_2 x ... x n_d).
        """
        return self.matmul(tf.eye(self.matrix_shape[0]))
