import pytest
import tensorflow as tf

from tool_tracking.tensor_train import TensorTrain


def test_tt_bad_shapes():
    with pytest.raises(ValueError, match=r"Invalid input and output shapes"):
        TensorTrain(
            input_shape=[4, 4, 4], output_shape=[3, 3], ranks=[1, 2, 2, 1]
        )


def test_tt_bad_ranks():
    with pytest.raises(
        ValueError, match=r"The ranks must have exactly one more value than"
    ):
        TensorTrain(
            input_shape=[4, 4, 4], output_shape=[3, 3, 3], ranks=[1, 2, 2, 2, 1]
        )

    with pytest.raises(
        ValueError, match=r"The first and the last rank must equal 1."
    ):
        TensorTrain(
            input_shape=[4, 4, 4], output_shape=[3, 3, 3], ranks=[2, 2, 2, 1]
        )


def test_matmul_bad_dim(random_tensor_train, random_matrix):

    tt = random_tensor_train
    X = random_matrix
    # mess up dimensions
    X = tf.reshape(X, [32, -1])

    with pytest.raises(
        ValueError, match=r"does not match the tensor train input shape"
    ):
        tt.matmul(X)


def test_matmul(random_tensor_train, random_matrix):

    tt = random_tensor_train
    X = random_matrix

    expected_result = tf.reshape(
        tf.einsum(
            "kabcd,aefw,bfgx,cghy,dhiz->kwxyz",
            # reshape X as tensor
            tf.reshape(X, [-1] + tt.input_shape),
            *[core for core in tt.cores]
        ),
        [-1, tt.matrix_shape[1]],
    )

    tf.debugging.assert_near(tt.matmul(X), expected_result, rtol=0.05)
