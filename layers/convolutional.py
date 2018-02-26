#!/usr/bin/env python3

import convutils
import numpy as np
import time

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
        self.bias = np.zeros((self.num_kernel, 1))
        self.padding = padding
        self.stride = stride
        self.kernels = np.ones( (num_kernel, input_shape[1] , kernel_size[1] , kernel_size[0]))
        self.output_shape = convutils.calculate_output_shape(input_shape[-1 : 1 : -1], kernel_size, padding, stride)
        #self.kernels = [np.random.random(self.output_shape) for i in range(num_kernel)]
        #k, i, j = convutils.get_im2col_indices(input_shape, kernel_size[0], kernel_size[1], padding, stride)
        X = np.random.randint(1, 4, input_shape )
        print("Input shape : {}".format(X.shape))
        print("Kernel shape : {}".format(kernel_size))
        print("Num Kernel: {}".format(num_kernel))
        print("Padding : {}".format(padding))
        print("Stride: {}".format(stride))
        print("Output shape : {}".format(self.output_shape))
        start = time.time()
        cols = convutils.im2col_indices(X,  kernel_size[0], kernel_size[1], padding, stride)
        #print(cols.shape)
        #print(X)
        #print(cols)
        wrow = self.kernels.reshape(self.num_kernel, -1)
        #print(wrow.shape)
        out = wrow @ cols
        # print(out.shape)
        out = out.reshape(self.num_kernel, self.output_shape[1], self.output_shape[0], self.input_shape[0])
        out = out.transpose(3, 0, 1, 2)
        print(time.time() - start)
        print(out.shape)


    def feed_forward(self):
        pass


def main():
    pass


if __name__ == "__main__":
    main()






