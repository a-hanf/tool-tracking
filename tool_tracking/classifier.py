from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention

import mlflow
import mlflow.tensorflow
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from os.path import exists
import numpy as np
import os
import argparse
from google_drive_downloader import GoogleDriveDownloader as gdd

parser = argparse.ArgumentParser(description='Provide the hyperparameters')
parser.add_argument('--model', type=str, default="lstm", help='Kind of Model to be trained')
parser.add_argument('--epochs', type=int, default=10, help='# epochs')
parser.add_argument('--bs', type=int, default=64, help='Train and eval batch size')
# parser.add_argument('--num_layers', type=int, default=1, help='Kind of Model to be trained')
parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[20], help='Hidden layer sizes')
parser.add_argument('--seed', type=int, default=424242, help='Tensorflow seed')

args = parser.parse_args()
model_choice = args.model
epochs = args.epochs
batch_size = args.bs
hidden_layers = args.hidden_sizes
num_layers = len(hidden_layers)
tf.random.set_seed(args.seed)
model_name = f'{model_choice}_{str(num_layers)}'

if not exists("../X_windowed.pickle"):
    print('Run data-preprocessing.py first in order to create the pickle files for the training data')
X = pickle.load(open('../X_windowed.pickle', 'rb'))
y = pickle.load(open('../y_windowed.pickle', 'rb'))


def download_from_google_drive(file_id, path):
    """Downloading data from Google drive"""
    gdd.download_file_from_google_drive(file_id=file_id, dest_path=path+"t",
                                        unzip=True, showsize=True, overwrite=True)
    # os.remove(path+"t")


path = "../pickles/"
file_id="1ZzzerpMMIjMFezSOhkmfTKZJCVqV5Pkh"
download_from_google_drive(file_id, "../pickles/") if not exists(path) else None
if not exists(path+"X_winsize_100.pickle"):
    print('Check file_id')

one_hot_bool = True
# X = pickle.load(open('../X_windowed.pickle', 'rb'))
# y = pickle.load(open('../y_windowed.pickle', 'rb'))

X = pickle.load(open(path+'X_winsize_100.pickle', 'rb'))
y = pickle.load(open(path+'y_winsize_100.pickle', 'rb'))
num_classes = len(np.unique(y))

# if one_hot_bool:
y = y.reshape(-1, 1)
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
y = enc.transform(y).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# uri = 'mysql://{user}:{password}@{hostname}:{port}/{databse}'
uri = f'file://{os.getcwd()}/mlruns/'
mlflow.set_tracking_uri(uri)


layer = input_layer = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
for layer_num in range(num_layers - 1):
    if "lstm" in model_choice:
        if "bi" in model_choice:
            layer = layers.Bidirectional(layers.LSTM(hidden_layers[layer_num], return_sequences=True))(layer)
        else:
            layer = layers.LSTM(hidden_layers[layer_num], return_sequences=True)(layer)
    elif "rnn" in model_choice:
        layer = layers.SimpleRNN(hidden_layers[layer_num], return_sequences=True)(layer)
    else:
        new_layer = None
if "attn" in model_choice:
    if "bi" in model_choice:
        layer_lstm = layers.Bidirectional(layers.LSTM(hidden_layers[-1], return_sequences=True))(layer)
    else:
        layer_lstm = layers.LSTM(hidden_layers[-1], return_sequences=True)(layer)
    layer_attn = SeqWeightedAttention()(layer_lstm)
    layer = layers.Add()([layer_lstm[:, -1, :], layer_attn])
else:
    if "bi" in model_choice:
        layer = layers.Bidirectional(layers.LSTM(hidden_layers[-1], return_sequences=False))(layer)
    else:
        layer = layers.LSTM(hidden_layers[-1], return_sequences=False)(layer)

# layer = recurrent_layer(layer, hidden_layers[-1], False, model_choice)
output = layers.Dense(num_classes, activation="softmax")(layer)
model = keras.Model(input_layer, output, name=model_name)

# model = keras.models.Sequential()
# model.add(keras.Input(shape=(X_train.shape[1], X_train.shape[2])))
# for layer_num in range(num_layers - 1):
#     if "lstm" in model_choice:
#         model.add(layers.LSTM(hidden_layers[layer_num], return_sequences=True))
#     elif "rnn" in model_choice:
#         model.add(layers.SimpleRNN(hidden_layers[layer_num], return_sequences=True))
#     else:
#         raise Exception
# layer_lstm = layers.LSTM(hidden_layers[-1], return_sequences=False)
# # model.add(SeqWeightedAttention())
# # model.add(layers.Bidirectional(layers.LSTM(hidden_layers[-1], return_sequences=True)))
# model.add(layers.Dense(num_classes, activation="softmax"))

# lstm, attn, lstm 68%

run_name = model_choice
with mlflow.start_run(run_name=run_name):
    # mlflow.tensorflow.autolog(every_n_iter=2)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam",
                  metrics=[tf.keras.metrics.categorical_accuracy,
                           tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(),
                           tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives()]
                  )
    model.summary()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    results = model.evaluate(X_test, y_test, batch_size=batch_size)
    print("test loss, test acc:", results)
    _, _, TP, FP, FN, TN = results
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + FP + FN + TN)
    # print("prec, rec, f1, acc", precision, recall, f1, acc_)

    mlflow.log_param("hidden sizes", hidden_layers)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch size", batch_size)
    mlflow.log_param("seed", args.seed)
    mlflow.log_metric("loss", results[0])
    mlflow.log_metric("categ acc", results[1])
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("TP", TP)
    mlflow.log_metric("FP", FP)
    mlflow.log_metric("FN", FN)
    mlflow.log_metric("TN", TN)



# else:
#     model.add(layers.Dense(1, activation="sigmoid"))
#     metrics = ["sparse_categorical_accuracy"]
#     loss = keras.losses.sparse_categorical_crossentropy
