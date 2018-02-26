#!/usr/bin/env python3

import numpy as np


def is_valid_stride(input_dim, kernel_dim, padding, stride):
    # output width
    xout = (input_dim[1] - kernel_dim[1] + 2*padding) / (stride[1]) + 1
    # output height
    yout = (input_dim[0] - kernel_dim[0] + 2*padding) / (stride[0]) + 1
    return xout.is_integer() and yout.is_integer()


def calculate_output_shape(input_shape, kernel_shape, padding, stride):
    # output width
    w = (input_shape[1] - kernel_shape[1] + 2*padding) / (stride[1]) + 1
    # output height
    h = (input_shape[0] - kernel_shape[0] + 2*padding) / (stride[0]) + 1
    return (int(h), int(w))


def get_im2col_indices(x_shape,
                       field_height=3, field_width=3,
                       padding=1, stride=(1, 1)
                    ):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride[0] == 0
    assert (W + 2 * padding - field_height) % stride[1] == 0
    out_height = (H + 2 * padding - field_height) / stride[0] + 1
    out_width = (W + 2 * padding - field_width) / stride[1] + 1
    print(out_height, out_width)

    i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)
    i0 = np.tile(i0, C)
    print("i0 :: {}".format(i0))

    i1 = stride[0] * np.repeat(np.arange(out_height, dtype='int32'), out_width)
    print("i1 :: {}".format(i1))

    j0 = np.tile(np.arange(field_width), field_height * C)
    print("j0:: {}".format(j0))

    j1 = stride[1] * np.tile(np.arange(out_width, dtype='int32'), int(out_height))
    print("j1:: {}".format(j1))

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C, dtype='int32'), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def main():
    pass

if __name__ == "__main__":
    main()

