import json
import pickle

import click

import sklearn.linear_model


@click.command()
@click.argument("name")
@click.argument("hid", type=int)
@click.option(
    "--solver",
    default="auto",
    type=click.Choice(["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
)
@click.option("--alpha", default=1.0, type=float)
def main(name, hid, solver, alpha):
    """Training & Testing Function"""
    model = sklearn.linear_model.Ridge(alpha=alpha, solver=solver)

    # training
    with open("./dataset/train.pkl", "rb") as f:
        (X, y) = pickle.load(f)
        model.fit(X, y)

    # testing
    with open("./dataset/test.pkl", "rb") as f:
        (X, y) = pickle.load(f)
        score = model.score(X, y)
        print(json.dumps({"metric": "score", "value": float(score)}))


if __name__ == "__main__":
    main()
