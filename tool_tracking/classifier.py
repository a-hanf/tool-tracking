from tensorflow import keras
from tensorflow.keras import layers
import pickle
model = keras.Sequential()
# do we need an embedding layer? our input is already in the "right" shape
# model.add(layers.Embedding(input_dim=11, output_dim=1, input_length=1))
model.add(layers.LSTM(20, return_sequences=True))
# didn't manage to add in a proper softmax layer
model.add(layers.Dense(1))
# SparseCategorical is for integer-encoded labels, Categorical expects 1-hot encoding
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer="sgd", metrics=["accuracy"])

try:
    X = pickle.load(open('../X_windowed.pickle', 'rb'))
    y = pickle.load(open('../y_windowed.pickle', 'rb'))
except FileNotFoundError:
    print('Run data-preprocessing.py first in order to create the pickle files for the training data')

print(X.shape)
print(y.shape)

model.fit(X, y, batch_size=20, epochs=10)
model.summary()


