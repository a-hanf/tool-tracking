from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import one_hot
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from os.path import exists
import numpy as np


if not exists("../X_windowed.pickle"):
    print('Run data-preprocessing.py first in order to create the pickle files for the training data')
one_hot_bool = True

X = pickle.load(open('../X_windowed.pickle', 'rb'))
y = pickle.load(open('../y_windowed.pickle', 'rb'))
num_classes = len(np.unique(y))

if one_hot_bool:
    y = y.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y)
    y = enc.transform(y).toarray()
    # y = one_hot(y, num_classes)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = keras.Sequential()
# do we need an embedding layer? our input is already in the "right" shape
# model.add(layers.Embedding(input_dim=11, output_dim=1, input_length=1))
model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(layers.LSTM(50, return_sequences=False))
# didn't manage to add in a proper softmax layer
if one_hot_bool:
    model.add(layers.Dense(num_classes, activation="softmax"))
else:
    model.add(layers.Dense(1, activation="sigmoid"))


# SparseCategorical is for integer-encoded labels, Categorical expects 1-hot encoding
if one_hot_bool:
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="Adam",
                  metrics=["categorical_accuracy"])
else:
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer="Adam",
                  metrics=["sparse_categorical_accuracy"])


print(X.shape)
print(y.shape)

model.fit(X_train, y_train, batch_size=64, epochs=50)
model.summary()

results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)
# Generate predictions (probabilities -- the output of the last layer) on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(X_test[:3])
print("predictions shape:", predictions.shape)



