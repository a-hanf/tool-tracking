from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from os.path import exists
import numpy as np
from urllib.parse import urlparse
import os
import argparse

from models import *


parser = argparse.ArgumentParser(description='Provide the hyperparameters')
parser.add_argument('--epochs', type=int, default=10, help='# epochs')
parser.add_argument('--bs', type=int, default=64, help='Train and eval batch size')
args = parser.parse_args()
epochs = args.epochs
batch_size = args.bs
one_hot_bool = True

if not exists("../X_windowed.pickle"):
    print('Run data-preprocessing.py first in order to create the pickle files for the training data')
X = pickle.load(open('../X_windowed.pickle', 'rb'))
y = pickle.load(open('../y_windowed.pickle', 'rb'))
num_classes = len(np.unique(y))

if one_hot_bool:
    y = y.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y)
    y = enc.transform(y).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# uri = 'mysql://{user}:{password}@{hostname}:{port}/{databse}'
uri = f'file://{os.getcwd()}/mlruns/'
mlflow.set_tracking_uri(uri)


model = keras.Sequential()
model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
# model = lstm_two_layer_attn(model)
model = lstm_two_layer(model)
# model = lstm_one_layer(model)
# model = rnn_one_layer(model)
if one_hot_bool:
    model.add(layers.Dense(num_classes, activation="softmax"))
    metrics = ["categorical_accuracy"]
    loss = keras.losses.categorical_crossentropy
else:
    model.add(layers.Dense(1, activation="sigmoid"))
    metrics = ["sparse_categorical_accuracy"]
    loss = keras.losses.sparse_categorical_crossentropy


with mlflow.start_run():
    # mlflow.tensorflow.autolog(every_n_iter=2)
    model.compile(loss=loss, optimizer="Adam", metrics=metrics)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    model.summary()

    results = model.evaluate(X_test, y_test, batch_size=batch_size)
    print("test loss, test acc:", results)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch size", batch_size)
    mlflow.log_metric("loss", results[0])
    mlflow.log_metric("categ acc", results[1])

    mlflow.log_metric("aaaa", results[1])
    feat_specifications = {
        "Acc": tf.Variable([], dtype=tf.int64, name="SepalLength"),
        "epochs": tf.Variable([], dtype=tf.float64, name="SepalWidth"),
        "PetalLength": tf.Variable([], dtype=tf.float64, name="PetalLength"),
        "PetalWidth": tf.Variable([], dtype=tf.float64, name="PetalWidth"),
    }

    # receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_specifications)
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        {'MY_FEATURE': tf.constant(2.0, shape=[1, 1])})
    saved_estimator_path = model.export_saved_model("model_file", receiver_fn).decode("utf-8")
# mlflow.tensorflow.save_model(model, "Models_trained/")
    # mlflow.log_artifacts("1", "mlflow/artifact")
    # mlflow.tensorflow.log_model(tf_saved_model_dir="runs:/1/run-relative/model", registered_model_name="mymodel")
    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    # if tracking_url_type_store != "file":
    #     mlflow.tensorflow.log_model(model, "model", registered_model_name="lstm_2")
    # else:
    #     mlflow.tensorflow.log_model(model, "model")
    # mlflow.tensorflow.load_model("models:/mymodel/1")

# Generate predictions (probabilities -- the output of the last layer) on new data using `predict`

# print("Generate predictions for 3 samples")
# predictions = model.predict(X_test[:3])
# print("predictions shape:", predictions.shape)



# tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
# if tracking_url_type_store != "file":
#     # Register the model
#     # There are other ways to use the Model Registry, which depends on the use case,
#     # please refer to the doc for more information:
#     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
#     mlflow.tensorflow.log_model(model, "model", registered_model_name="lstm2", tf_meta_graph_tags=[tag_constants.SERVING])
# else:
#     mlflow.tensorflow.log_model(model, "model")


