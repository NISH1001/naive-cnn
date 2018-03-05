#!/usr/bin/env python3

import convutils
import numpy as np
import time

from convolution import convolve
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
                input_shape:    shape of individual input instance (channels, height, width)
                kernel_size:   shape of the kernel for  performing convolution (height, width)
                num_kernel:     number of kernles (number  of output channel it is)
                padding:        number of boundary layer to pad the input (0 means no padding)
                stride:         the step in both direction to take while performing convolution
        """

        hout, wout = convutils.calculate_output_shape(input_shape[1:], kernel_size, padding, stride)
        self.output_shape = num_kernel, hout, wout
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.num_kernel = num_kernel
        self.padding = padding
        self.stride = stride

        # each row of this 4d matrix is a kernel with depth spanning to input channels
        self.kernels = np.random.randint(1, 8, (num_kernel, input_shape[0], kernel_size[0], kernel_size[1]))
        self.bias = np.zeros((self.num_kernel, 1))

    def feed_forward(self, X):
        print("X::\n{}".format(X))
        self.X = X
        self.X_cols = convutils.im2col_indices(X, self.kernel_size[0], self.kernel_size[1], self.padding, self.stride)
        print("self.X_cols::\n{}".format(self.X_cols))
        print("self.X_cols shape :: {}".format(self.X_cols.shape))

        wcol = self.kernels.reshape(self.num_kernel, -1)
        print("wcol shape:: {}".format(wcol.shape))

        out = wcol @ self.X_cols + self.bias
        #out = np.dot(wcol, self.X_cols) + self.bias
        print(out.shape)
        out = out.reshape(self.num_kernel, self.output_shape[1], self.output_shape[2], X.shape[0])
        out = out.transpose(3, 0, 1, 2)
        return out

    def backpropagate(self, delta):
        delta_bias = np.sum(delta, axis=(0, 2, 3)).reshape(self.num_kernel, -1)

        delta_flat = delta.transpose(1, 2, 3, 0).reshape(self.num_kernel, -1)
        delta_kernel = delta_flat @ self.X_cols.T
        delta_kernel = delta_kernel.reshape(self.kernels.shape)

        kernels_flat = self.kernels.reshape(self.num_kernel, -1)
        delta_Xcol = kernels_flat.T @ delta_flat
        dX = convutils.col2im_indices(delta_Xcol, self.X.shape,
                                      self.kernel_size[0],
                                      self.kernel_size[1],
                                      self.padding,
                                      self.stride
                                    )
        return dX, delta_kernel, delta_bias

    def __repr__(self):
        string = "Individual Input Shape :: {}\nKernel Shape :: {}\nPadding :: {}\nStride :: {}\nOutput Shape :: {}".format(
            self.input_shape,
            self.kernel_size,
            self.padding,
            self.stride,
            (self.output_shape)
        )
        return string


class ConvMono(Layer):
    def __init__(self, input_shape, kernel_size,
                 padding=1, stride=(1, 1)):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.num_kernel = 1
        hout, wout = convutils.calculate_output_shape(input_shape, kernel_size, padding, stride)
        self.output_shape = (hout, wout)
        self.kernel = np.random.random(kernel_size)
        self.bias = np.zeros((self.num_kernel, 1))
        self.padding = padding
        self.stride = stride

    def feed_forward(self, X):
        self.X = X
        out = convolve(X, self.kernel, padding=self.padding, stride=self.stride) + self.bias
        return out

    def backpropagate(self, dout):
        dX = np.zeros_like(self.X)
        dW = np.zeros_like(self.kernel)
        h, w = dout.shape
        kh, kw = self.kernel.shape
        for r in range(0, h, self.stride[0]):
            for c in range(0, w, self.stride[1]):
                dX[r : r + kh, c : c + kw ] += self.kernel * dout[(r, c)]
                dW += self.X[r : r + kh, c : c + kw] * dout[(r, c)]
        return dX, dW


def main():
    pass


if __name__ == "__main__":
    main()
