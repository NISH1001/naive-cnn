#!/usr/bin/env python3

import numpy as np
from layers.layer import Layer


class Flatten(Layer):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.output_shape = (np.prod(self.input_shape), )
        self.params = []

    def feed_forward(self, X):
        output_shape = (X.shape[0], -1)
        out = X.ravel().reshape(output_shape)
        return out

    def backpropagate(self, grad_out):
        return grad_out.reshape(self.input_shape), []

class Dense(Layer):
    def __init__(self, neurons, input_shape=None):
        self.neurons = neurons
        self.input_shape = input_shape
        self.output_shape =(self.neurons, )
        self.synapse = np.random.random((self.input_shape[0], neurons))
        self.bias = np.random.random((1, neurons))

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
