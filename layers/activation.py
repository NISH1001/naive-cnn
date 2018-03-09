#!/usr/bin/env python3

import numpy as np

from layers.layer import Layer

class Sigmoid(Layer):
    def __init__(self):
        pass

    def feed_forward(self, X):
        out = 1/(1 + np.exp(-X))
        self.out = out
        return out

    def backpropagate(self, dout):
        dX = dout * self.out * (1 - self.out)
        return dX, []


def main():
    pass


if __name__ == "__main__":
    main()
