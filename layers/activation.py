#!/usr/bin/env python3

import numpy as np

from layers.layer import Layer

class Sigmoid(Layer):
    def __init__(self):
        pass

    def feed_forward(self, X):
        return 1/(1 + np.exp(-X))

    def backpropagate(self, dout):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
