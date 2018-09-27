"""

"""

import numpy as np


class Sigmoid:

    cache = None

    def __init__(self):
        pass

    def __call__(self, x):
        self.cache = x
        # TODO: Sigmoid Computation
        return x

    def backward(self, grad_in):
        x = self.cache
        # TODO: Gradient Computation
        return x


class ReLU:

    cache = None

    def __init__(self):
        pass

    def __call__(self, x):
        self.cache = x
        # TODO: Sigmoid Computation
        return x

    def backward(self, grad_in):
        x = self.cache
        # TODO: Gradient Computation
        return x


class LeakyReLU:
    cache = None
    negative_slope = 1e-2

    def __init__(self, negative_slope):
        self.negative_slope = negative_slope

    def __call__(self, x):
        self.cache = x
        # TODO: Sigmoid Computation
        return x

    def backward(self, grad_in):
        x = self.cache
        # TODO: Gradient Computation
        return x


class Softmax:

    cache = None

    def __init__(self):
        pass

    def __call__(self, x):
        self.cache = x
        # TODO: Sigmoid Computation
        return x

    def backward(self, grad_in):
        x = self.cache
        # TODO: Gradient Computation
        return x


class Tanh:
    cache = None

    def __init__(self):
        pass

    def __call__(self, x):
        self.cache = x
        # TODO: Sigmoid Computation
        return x

    def backward(self, grad_in):
        x = self.cache
        # TODO: Gradient Computation
        return x


class ELU:

    cache = None

    def __init__(self):
        pass

    def __call__(self, x):
        self.cache = x
        # TODO: Sigmoid Computation
        return x

    def backward(self, grad_in):
        x = self.cache
        # TODO: Gradient Computation
        return x
