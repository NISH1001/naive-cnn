#!/usr/bin/env python3

import numpy as np

class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self):
        for layer in self.layers[1:]:
            print(layer)

    def fit(self, X, Y):
        predicted = self.feed_forward(X)
        return predicted

    def feed_forward(self, X):
        inp = X
        for layer in self.layers:
            inp = layer.feed_forward(inp)
        return inp


    def loss_der(self, predicted, target):
        return (predicted-target)
