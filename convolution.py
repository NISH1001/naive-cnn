#!/usr/bin/env python3

import convutils
import numpy as np


def im2col(input, kernel_shape, channel=1):
    wout = input.shape[1] - kernel_shape[1] + 1
    hout = input.shape[0] - kernel_shape[0] + 1
    res = np.array((hout*wout, kernel_shape[0] * kernel_shape[1] * channel))
    for r in range(hout):
        for c in range(wout):
            input_sliced = input[r: r + kernel_shape[0], c: c + kernel_shape[1]]
    return res


def convolve3d(input, kernel, depth=1, padding=0, stride=(1,1)):
    pass


def convolve(data, kernel, padding=0, stride=(1, 1)):
    """
        A simple convolution function.
        For now the code just works like any other convolution.
        There is a lot of room for optimization.

        Args:
            data:    This is the input matrix/array.
            kernel: This is the convolution filter/kernel to be applied
            stride: This is how much the kernel is shifted with each step

        Returns:
            A 2d matrix/array after applying convolution

        Raises:
            ValueError: Raises exception when dimensions are mismatched between input and the kernel
    """
    shape_input = data.shape
    shape_kernel = kernel.shape
    hout, wout = convutils.calculate_output_shape(shape_input, shape_kernel, padding, stride)
    result = np.zeros((hout, wout))

    h_padded, w_padded = shape_input[0] + 2 * padding, shape_input[1] + 2 * padding
    if padding:
        data = np.pad(data, pad_width=padding, mode='constant')

    # for each row apply convolution accordingly
    for r in range(0, hout, stride[0]):
        for c in range(0, wout, stride[1]):
            input_sliced = data[r : r + shape_kernel[0], c: c + shape_kernel[1]]
            result[(r, c)] = np.sum(input_sliced * kernel)
    return result


def main():
    # x = np.random.random( (5, 5) )
    x = np.zeros((5, 5))
    x[0][0] = 1
    x[2][0] = 1
    kernel = np.ones((3, 3))
    result = convolve(x, kernel)
    print(result)


if __name__ == "__main__":
    main()
