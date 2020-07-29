from math import sqrt

eps = 1e-6
delta = 1e-6
MAX_ITER = 500


def df(f, x):
    return (f(x) - f(x - delta)) / delta


class Optimizer:

    @classmethod
    def get_optimizer(cls, name):
        return next(cl for cl in cls.__subclasses__() if cl.name == name)()

    def run(self, x_0, f, a, **kwargs):
        x_k = x_0
        i = 0
        path = [x_0]
        while abs(df(f, x_k) / df(f, x_0)) > eps and i < MAX_ITER:
            x_k = self.optimize(x_k, f, a=a, **kwargs)
            path.append(x_k)
            i = i + 1

        print(f"It took {i} iterations to converge!", flush=True)
        print(f"The optimal value occurs at {round(x_k, 2)} where the value of the function is {round(f(x_k), 2)}")
        return path

    def optimize(self, x_k, f, a, **kwargs):
        pass


class SGD(Optimizer):
    name = 'sgd'

    def optimize(self, x_k, f, a,  **kwargs):
        x_k = x_k - a * df(f, x_k)
        return x_k


class Momentum(Optimizer):
    name = 'momentum'

    def optimize(self, x_k, f, a, **kwargs):
        d_k = getattr(self, "d_k", df(f, x_k))
        B = kwargs.get('B', 0.9)
        d_k = B * d_k + (1 - B) * df(f, x_k)
        x_k = x_k - a * d_k
        self.d_k = d_k
        return x_k


class RMSProp(Optimizer):
    name = 'rms_prop'

    def optimize(self, x_k, f, a, **kwargs):
        B = kwargs.get('B', 0.99)
        d_k = getattr(self, 'd_k', df(f, x_k) ** 2)
        d_k = B * d_k + (1 - B) * (df(f, x_k) ** 2)
        x_k = x_k - a * df(f, x_k) / sqrt(d_k + eps)
        return x_k


class Adam(Optimizer):
    name = 'adam'

    def optimize(self, x_k, f, a, **kwargs):
        B_1 = kwargs.get('B_1', 0.99)
        B_2 = kwargs.get('B_2', 0.999)
        m_k = getattr(self, 'm_k', df(f, x_k))
        d_k = getattr(self, 'd_k', df(f, x_k)**2)
        m_k = B_1 * m_k + (1 - B_1) * df(f, x_k)
        d_k = B_2 * d_k + (1 - B_2) * (df(f, x_k) ** 2)
        x_k = x_k - a * m_k / sqrt(d_k + eps)
        self.m_k = m_k
        self.d_k = d_k
        return x_k


class ConjugateGradient(Optimizer):
    name = 'conjugate_gradient'

    def optimize(self, x_k, f, a, **kwargs):
        d_k = getattr(self, 'd_k', df(f, x_k))
        prev_grad = df(f, x_k)
        x_k = x_k - a * d_k
        B_k = df(f, x_k) * df(f, x_k) / (prev_grad * prev_grad)
        d_k = df(f, x_k) - B_k * d_k
        self.d_k = d_k
        return x_k
