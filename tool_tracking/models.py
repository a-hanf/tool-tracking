from tensorflow.keras import layers

from tool_tracking.tensor_train import TensorTrain
from tool_tracking.layers import TTDense, TTLSTMCell


def lstm_two_layer_attn(model):
    model.add(layers.LSTM(100, return_sequences=True))
    model.add(layers.LSTM(50, return_sequences=False))
    model.add(layers.Attention())
    return model


def lstm_two_layer(model):
    model.add(layers.LSTM(100, return_sequences=True))
    model.add(layers.LSTM(50, return_sequences=False))
    return model


def tt_lstm_two_layer(model, rank=4):
    model.add(
        layers.RNN(
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
        layers.RNN(
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


def lstm_one_layer(model):
    model.add(layers.LSTM(100, return_sequences=False))
    return model


def rnn_one_layer(model):
    model.add(layers.SimpleRNN(50, return_sequences=False))
    return model
