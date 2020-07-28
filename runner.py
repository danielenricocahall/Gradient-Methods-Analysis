import os

import numpy as np
import matplotlib.pyplot as plt

import matplotlib

from optimizer import Optimizer

eps = 1e-3
delta = 1e-6
MAX_ITER = 500


def f(x):
    return np.square(x) / 10 - 2 * np.sin(x)


def df(x):
    return (f(x) - f(x - delta)) / delta


def main():
    x_0 = -13
    a = 0.01

    opt = Optimizer.get_optimizer('conjugate_gradient')
    path = opt.run(x_0, f, a)
    foo, bar = zip(*path)
    x_f = foo[-1]

    x = np.linspace(-20, 20, 200)
    plt.plot(x, f(x))
    plt.plot(foo, bar, label='f($x_k$)')
    plt.plot(x_f, f(x_f), 'ro')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    exit()
