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


def tt_lstm_two_layers_1(model):
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(
        layers.RNN(
            TTLSTMCell(
                tt=TensorTrain(
                    input_shape=[4, 4, 4],
                    output_shape=[4, 8, 4],
                    ranks=[1, 4, 4, 1],
                ),
                activation="tanh",
                kernel_initializer="glorot_normal",
            )
        )
    )

    return model


def tt_lstm_two_layer_2(model):
    model.add(
        layers.RNN(
            TTLSTMCell(
                tt=TensorTrain(
                    input_shape=[11, 1, 1, 1],
                    output_shape=[4, 4, 4, 4],
                    ranks=[1, 4, 4, 4, 1],
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
                    input_shape=[4, 4, 4],
                    output_shape=[4, 8, 4],
                    ranks=[1, 4, 4, 1],
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
