#!/usr/bin/env python3

import numpy as np
np.random.seed(0)

def convolve(inp, kernel, stride=(1, 1)):
    """
        A simple convolution function.

        Args:
            inp:    This is the input matrix/array.
            kernel: This is the convolution filter/kernel to be applied
            stride: This is how much the kernel is shifted with each step

        Returns:
            A 2d matrix/array after applying convolution

        Raises:
            ValueError: Raises exception when dimensions are mismatched between input and the kernel
    """
    shape_inp = inp.shape
    shape_kernel = kernel.shape

    if (shape_inp[0] < shape_kernel[0]) or (shape_inp[1] < shape_kernel[1]):
        raise ValueError("Invalid dimension between input and kernel...")

    # calculate the number of steps in both direction
    iterx = shape_inp[1] - shape_kernel[1] + 1
    itery = shape_inp[0] - shape_kernel[0] + 1

    result = np.zeros( (itery, iterx) )

    # for each row apply convolution accordingly
    for r in range(itery):
        for c in range(iterx):
            inp_sliced = inp[r : r + shape_kernel[0], c : c + shape_kernel[1]]
            res = np.sum(inp_sliced * kernel)
            result[ (r, c) ] = res
    return result

def main():
    #x = np.random.random( (5, 5) )
    x = np.zeros( (5, 5) )
    x[0][0] = 1
    x[2][0] = 1
    kernel = np.ones( (3, 3) )
    result = convolve(x, kernel)
    print(result)

if __name__ == "__main__":
    main()

