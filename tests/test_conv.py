#!/usr/bin/env python3

import numpy as np

from layers.convolutional import Conv2D


def run():
    input_shape = (2, 5, 5)
    X = np.random.randint(1, 4, (3, ) + input_shape)
    conv2d = Conv2D(
        input_shape=input_shape,
        kernel_size=(2, 2),
        num_kernel=3,
        padding=0,
        stride=(1, 1)
    )
    print(conv2d)
    out = conv2d.feed_forward(X)
    conv2d.backpropagate(out)
    print("After conv2d, shape :: {}".format(out.shape))


def main():
    pass


if __name__ == "__main__":
    main()

