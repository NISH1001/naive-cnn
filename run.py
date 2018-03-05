#!/usr/bin/env python3

import numpy as np
import tests

np.random.seed(0)


def main():
     tests.test_conv.run()
    # tests.test_convolution.run()


if __name__ == "__main__":
    main()
