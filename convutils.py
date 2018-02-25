#!/usr/bin/env python3

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
    return (h, w)


def main():
    pass

if __name__ == "__main__":
    main()

