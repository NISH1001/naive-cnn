#!/usr/bin/env python3

def load_mnist_train(filename):
    print("Loading training data...")
    X_train = []
    Y_train = []
    with open(filename, 'r') as f:
        for line in f:
            splitted = line.split(',')
            try:
                label = int(splitted[0])
                vals = splitted[1:]
                Y_train.append(label)
                X_train.append(list(map(int, vals)))
            except ValueError:
                continue
    return X_train, Y_train

def load_mnist_test(filename):
    print("Loading test data...")
    X_test = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            vals = line.split(',')
            try:
                X_test.append(list(map(int, vals)))
            except ValueError:
                continue
    return X_test


def main():
    pass

if __name__ == "__main__":
    main()

