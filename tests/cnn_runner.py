#!/usr/bin/env python3


import dataloader
import numpy as np

from layers.convolutional import Conv2D, ConvMono
from layers.activation import Sigmoid
from layers.misc import Flatten, Dense

from model import Model

def test():
    # input_shape = (5, 5)
    # X = np.random.random(input_shape)
    # convmono = ConvMono(
    #     input_shape = input_shape,
    #     kernel_size = (3, 3),
    #     padding = 0,
    #     stride = (1, 1)
    # )

    input_shape = (2, 5, 5)
    num_input = 5
    X = np.random.randint(1, 4, (num_input, ) + input_shape)
    c1 = Conv2D(
        input_shape=input_shape,
        kernel_size=(2, 2),
        num_kernel=3,
        padding=0,
        stride=(1, 1)
    )

    c2 = Conv2D(
        input_shape=c1.output_shape,
        kernel_size=(2, 2),
        num_kernel=4,
        padding=0,
        stride=(1, 1)
    )

    sigmoid = Sigmoid(input_shape=c2.output_shape)
    flatten = Flatten(input_shape=sigmoid.output_shape)
    h1 = Dense(neurons=25, input_shape=flatten.output_shape)
    h2 = Dense(neurons=12, input_shape=h1.output_shape)
    output = Dense(neurons=2, input_shape=h2.output_shape)


    model = Model()
    model.add_layer(c1)
    model.add_layer(c2)
    model.add_layer(sigmoid)
    model.add_layer(flatten)
    model.add_layer(h1)
    model.add_layer(h2)
    model.add_layer(output)
    model.compile()

    y = model.fit(X, X)
    print(y.shape)


def test_mnist():
    images, labels = dataloader.load_mnist_train("data/train.csv")
    print(np.array(images).shape)

def main():
    test_mnist()

if __name__ == "__main__":
    main()
