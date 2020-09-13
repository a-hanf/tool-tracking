from tensorflow.keras import layers as keras_layers
from tensor_train import TensorTrain
from layers import TTDense, TTLSTMCell



def tt_lstm_two_layer(model, rank=4):
    model.add(
        keras_layers.RNN(
            TTLSTMCell(
                tt=TensorTrain(
                    input_shape=[11, 1, 1, 1],
                    output_shape=[4, 4, 4, 2],
                    ranks=[1, rank, rank, rank, 1],
                ),
                activation="tanh",
                kernel_initializer="glorot_normal",
            ),
            return_sequences=True,
        )
    )

    model.add(
        keras_layers.RNN(
            TTLSTMCell(
                tt=TensorTrain(
                    input_shape=[4, 4, 2],
                    output_shape=[4, 6, 4],
                    ranks=[1, rank, rank, 1],
                ),
                activation="tanh",
                kernel_initializer="glorot_normal",
            )
        )
    )
    return model
