#!/usr/bin/env python3

import numpy as np

from layers.layer import Layer

class Sigmoid(Layer):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.output_shape = self.input_shape

    def feed_forward(self, X):
        out = 1/(1 + np.exp(-X))
        self.out = out
        return out

    def backpropagate(self, dout):
        dX = dout * self.out * (1 - self.out)
        return dX, []

    def __call__(self, X):
        return 1/(1 + np.exp(-X))


def main():
    pass


if __name__ == "__main__":
    main()
