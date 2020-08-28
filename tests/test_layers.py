from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential

from tool_tracking.tensor_train import TensorTrain
from tool_tracking.layers import TTDense


def test_mnist():
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # standardize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    # add tensor train dense layer
    model.add(
        TTDense(
            TensorTrain(
                input_shape=[7, 4, 7, 4],
                output_shape=[4, 4, 4, 4],
                ranks=[1, 4, 4, 4, 1],
            ),
            activation="relu",
            bias_initializer=1e-3,
        )
    )
    model.add(Dense(10))
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(lr=1e-2),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    assert history.history["val_accuracy"][-1] > 0.95
