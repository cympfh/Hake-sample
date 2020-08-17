import json

import click

from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils


@click.command()
@click.argument("name")
@click.argument("hid", type=int)
def main(
    name: str,
    hid: int,
):
    """Testing Function"""
    _, (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("f") / 255.0
    y_test = np_utils.to_categorical(y_test, 10)

    model = load_model(f"./models/{name}_{hid:08d}.h5")
    _, acc = model.evaluate(x_test, y_test, batch_size=128)

    print(json.dumps({"metric": "acc", "value": float(acc)}))


if __name__ == "__main__":
    main()
