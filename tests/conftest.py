# pylint: skip-file
"""Test configuration."""

import numpy as np
import tensorflow as tf
import pytest

from tool_tracking.tensor_train import TensorTrain

INPUT_SHAPE = [4, 6, 8, 10]
OUTPUT_SHAPE = [3, 5, 7, 9]
RANKS = [1, 2, 2, 2, 1]


@pytest.fixture(scope="function")
def random_matrix(batch_size=16, input_shape=None):
    if input_shape is None:
        input_shape = np.prod(INPUT_SHAPE)

    return tf.random.normal((batch_size, input_shape))


@pytest.fixture(scope="function")
def random_tensor_train(input_shape=None, output_shape=None, ranks=None):

    if input_shape is None:
        input_shape = INPUT_SHAPE

    if output_shape is None:
        output_shape = OUTPUT_SHAPE

    if ranks is None:
        ranks = RANKS

    tt = TensorTrain(input_shape, output_shape, ranks)

    tt.init_cores(tf.random.normal)
    return tt
