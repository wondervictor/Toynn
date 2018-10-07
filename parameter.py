"""

"""
import numpy as np


class Parameter:

    def __init__(self, shape, with_grad=False, initializer=None):

        self.data = None
        self.grad = None
        self.initializer = initializer
        self.shape = shape

        self.initialize(with_grad=with_grad)

    def get_grad(self):
        return self.grad

    def set_grad(self, grad):
        self.grad = grad

    def get_data(self):
        return self.data

    def initialize(self, with_grad=False):
        self.data = self.initializer(self.shape)
        if with_grad:
            self.grad = np.zeros(self.shape)

    def update(self, delta):
        self.data += delta


class GaussianInitializer:

    def __init__(self, mean=0.0, std=1e-2):
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        return np.random.normal(self.mean, self.std, shape)


class ConstantInitializer:

    def __init__(self, constant=0):
        self.constant = constant

    def __call__(self, shape):
        return np.zeros(shape) + self.constant


class UniformInitializer:

    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high

    def __call__(self, shape):
        return np.random.uniform(self.low, self.high, shape)


class MSRAInitializer:

    def __init__(self):
        pass

    def __call__(self, shape):
        pass

