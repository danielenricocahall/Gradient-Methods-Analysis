import numpy as np
import matplotlib.pyplot as plt

from optimizer import Optimizer


def f(x):
    return np.square(x) / 10 - 2 * np.sin(x)


def main():
    x_0 = -13
    a = 0.01

    opt = Optimizer.get_optimizer('conjugate_gradient')
    x = opt.run(x_0, f, a)
    x_f = x[-1]

    plt.plot(np.linspace(-20, 20, 200), f(np.linspace(-20, 20, 200)))
    plt.plot(x, [f(x_k) for x_k in x], label='f($x_k$)')
    plt.plot(x_f, f(x_f), 'ro')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    exit()
