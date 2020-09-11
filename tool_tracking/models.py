from tensorflow.keras import layers


def lstm_two_layer_attn(model):
    model.add(layers.LSTM(100, return_sequences=True))
    model.add(layers.LSTM(50, return_sequences=False))
    model.add(layers.Attention())
    return model


def lstm_two_layer(model):
    model.add(layers.LSTM(100, return_sequences=True))
    model.add(layers.LSTM(50, return_sequences=False))
    return model


def lstm_one_layer(model):
    model.add(layers.LSTM(100, return_sequences=False))
    return model


def rnn_one_layer(model):
    model.add(layers.SimpleRNN(50, return_sequences=False))
    return model
