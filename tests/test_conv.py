#!/usr/bin/env python3

from layers.convolutional import Conv2D

def run():
    conv2d = Conv2D(input_shape=(32, 32), kernel_size=(3, 3), padding=1, stride=(1, 1) )
    print("Conv2D")

def main():
    pass

if __name__ == "__main__":
    main()

