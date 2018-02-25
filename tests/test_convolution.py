#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from convolution import convolve

BASEWIDTH = 256


def load_image(filename):
    return Image.open(filename)


def resize(img):
    # wpercent = (BASEWIDTH / float(img.size[0]))
    # hsize = int((float(img.size[1])*float(wpercent)))
    return img.resize((BASEWIDTH,BASEWIDTH), Image.ANTIALIAS)


def binarize(img):
    return img.convert('L')


def run():
    print("Testing convolution...")
    image = load_image("./data/bird.jpg")
    image = binarize(resize(image))
    data = np.asarray(image)

    # kernel = np.asarray([ [1, 0, -1], [0, 1, 0], [-1, 0, 1] ])
    # kernel = np.asarray([ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ])
    kernel_edge = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel_sharpen = np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel_blur = np.ones((3, 3)) / 9

    res = convolve(data, kernel_sharpen, padding=0, stride=(1, 1))
    image_res = Image.fromarray(res.astype('uint8'), 'L')
    image_res.save("./data/res.jpg")


def main():
    pass


if __name__ == "__main__":
    main()
