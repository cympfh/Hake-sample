import click

from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import np_utils


@click.command()
@click.argument("name")
@click.argument("hid", type=int)
@click.option("--epoch", type=int)
@click.option("--batch-size", type=int)
@click.option("--hidden-dim", type=int)
@click.option("--activation", type=click.Choice(["sigmoid", "relu"]))
@click.option("--optimizer", type=click.Choice(["sgd", "adam"]))
def main(
    name: str,
    hid: int,
    epoch: int,
    batch_size: int,
    hidden_dim: int,
    activation: str,
    optimizer: str,
):
    """Training Function"""
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype("f") / 255.0
    y_train = np_utils.to_categorical(y_train, 10)

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(hidden_dim, activation=activation))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch)

    model.save(f"./models/{name}_{hid:08d}.h5", save_format="h5")


if __name__ == "__main__":
    main()
