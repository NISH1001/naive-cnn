#!/usr/bin/env python3

import numpy as np

from layers.convolutional import Conv2D, ConvMono
from layers.activation import Sigmoid
from layers.misc import Flatten, Dense

from model import Model


def main():
    input_shape = (5, 5)
    X = np.random.random(input_shape)

    convmono = ConvMono(
        input_shape = input_shape,
        kernel_size = (3, 3),
        padding = 0,
        stride = (1, 1)
    )

    input_shape = (2, 5, 5)
    num_input = 5
    X = np.random.randint(1, 4, (num_input, ) + input_shape)
    conv2d = Conv2D(
        input_shape=input_shape,
        kernel_size=(2, 2),
        num_kernel=3,
        padding=0,
        stride=(1, 1)
    )

    sigmoid = Sigmoid()
    flatten = Flatten()

    model = Model()
    model.add_layer(conv2d)
    # model.add_layer(sigmoid)
    model.add_layer(flatten)
    model.compile()

    # y = model.fit(X, X)


if __name__ == "__main__":
    main()
