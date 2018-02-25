#!/usr/bin/env python3

from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def feed_forward(self):
        pass


class TestLayer(Layer):
    def feed_forward(self):
        pass



def main():
    t = TestLayer()

if __name__ == "__main__":
    main()

