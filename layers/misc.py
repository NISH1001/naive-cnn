#!/usr/bin/env python3

import numpy as np
from layers.layer import Layer


class Flatten(Layer):
    def __init__(self):
        self.params = []

    def feed_forward(self, X):
        self.input_shape = X.shape
        self.output_shape = (self.input_shape[0], -1)
        out = X.ravel().reshape(self.output_shape)
        self.output_shape = out.shape
        return out

    def backpropagate(self, grad_out):
        return grad_out.reshape(self.input_shape), []

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.synapse = np.random.random((input_size, output_size))
        self.bias = np.random.random((1, output_size))

    def feed_forward(self, X):
        self.X = X
        return np.dot(X, self.synapse) + self.bias

    def backpropagate(self, grad_out):
        grad_synapse = np.dot(self.X.T, grad_out)
        grad_bias = np.sum(grad_out, axis=0)
        grad_X = np.dot(grad_out, self.synapse.T)
        return grad_X, grad_synapse, grad_bias

def main():
    pass


if __name__ == "__main__":
    main()
