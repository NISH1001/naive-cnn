#!/usr/bin/env python3

import convutils
import numpy as np

from layers.layer import Layer


class Conv2D(Layer):
    """
        A 2D Convolutional Layer where convolution is applied to input data
        using the available kernels.
    """
    def __init__(self, input_shape, kernel_size, num_kernel=1,
                 padding=1, stride=(1, 1)):
        """
            Constructor for  Conv2D Layer

            Args:
                input_shape:    shape of individual input instance (height, width)
                kernel_size:   shape of the kernel for  performing convolution (height, width)
                num_kernel:     number of kernles (number  of output channel it is)
                padding:        number of boundary layer to pad the input (0 means no padding)
                stride:         the step in both direction to take while performing convolution
        """
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.num_kernel = num_kernel
        self.padding = padding
        self.stride = stride
        self.output_shape = convutils.calculate_output_shape(input_shape, kernel_size, padding, stride)
        print(self.output_shape)
        #self.kernels = [np.random.random(self.output_shape) for i in range(num_kernel)]

    def feed_forward(self):
        pass


def main():
    pass


if __name__ == "__main__":
    main()

