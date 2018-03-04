#!/usr/bin/env python3

import numpy as np


def is_valid_stride(input_dim, kernel_dim, padding, stride):
    """
        Check if stride is valid
    """
    # output width
    xout = (input_dim[1] - kernel_dim[1] + 2*padding) / (stride[1]) + 1
    # output height
    yout = (input_dim[0] - kernel_dim[0] + 2*padding) / (stride[0]) + 1
    return xout.is_integer() and yout.is_integer()


def calculate_output_shape(input_shape, kernel_shape, padding, stride):
    """
        Out = (Input - Kernel + 2*padding) / stride + 1

        which should be integer in both ways otherwise convolution will miss
        some portions of input pixels
    """
    w = (input_shape[1] - kernel_shape[1] + 2*padding) / (stride[1]) + 1
    h = (input_shape[0] - kernel_shape[0] + 2*padding) / (stride[0]) + 1
    if not h.is_integer() or not w.is_integer():
        raise ValueError("Non-integer output dimension. Make sure padding and strides are good")
    return int(h), int(w)


def get_im2col_indices(X_shape,
                       field_height=3, field_width=3,
                       padding=1, stride=(1, 1)):
    """
        Create fancy indices for 3d tensor
    """
    N, C, H, W = X_shape
    # assert (H + 2 * padding - field_height) % stride[0] == 0
    # assert (W + 2 * padding - field_height) % stride[1] == 0
    out_height = (H + 2 * padding - field_height) / stride[0] + 1
    out_width = (W + 2 * padding - field_width) / stride[1] + 1

    i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)
    i0 = np.tile(i0, C)

    i1 = stride[0] * np.repeat(np.arange(out_height, dtype='int32'), out_width)

    j0 = np.tile(np.arange(field_width), field_height * C)

    j1 = stride[1] * np.tile(np.arange(out_width, dtype='int32'), int(out_height))

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C, dtype='int32'), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(X, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(X.shape, field_height, field_width, padding, stride)
    cols = X_padded[:, k, i, j]
    C = X.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, X_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = X_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(X_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]


def main():
    pass


if __name__ == "__main__":
    main()
