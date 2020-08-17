# Generate Random Dummy Dataset

import numpy.random
import pickle


w = numpy.random.rand(10)
b = numpy.random.rand()

# train
x = numpy.random.rand(100, 10)
noise = numpy.random.randn(100)
y = x @ w + b + noise

with open("./dataset/train.pkl", "wb") as f:
    pickle.dump((x, y), f)

# test
x = numpy.random.rand(100, 10)
noise = numpy.random.randn(100)
y = x @ w + b + noise

with open("./dataset/test.pkl", "wb") as f:
    pickle.dump((x, y), f)
