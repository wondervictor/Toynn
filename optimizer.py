"""


"""

import numpy as np


class SGD:

    def __init__(self, lr):
        self.learning_rate = lr

    def set_lr(self, lr):
        self.learning_rate = lr

    def optimize(self, grad, name):
        delta = -grad * self.learning_rate
        return delta


class Momentum:

    learning_rate = 1e-2

    def __init__(self, lr):
        self.learning_rate = lr

    def set_lr(self, lr):
        self.learning_rate = lr

    def _optimize_func(self, data, grad):
        delta = 0.0 * self.learning_rate
        # TODO: Momentum Optimize
        return delta

    def optimize(self, params):

        for param in params:
            delta = self._optimize_func(param.get_data(), param.get_grad())
            param.update(delta)


class AdaDelta:

    learning_rate = 1e-2

    def __init__(self, lr):
        self.learning_rate = lr

    def set_lr(self, lr):
        self.learning_rate = lr

    def _optimize_func(self, data, grad):
        delta = 0.0 * self.learning_rate
        # TODO: AdaDelta Optimize
        return delta

    def optimize(self, params):

        for param in params:
            delta = self._optimize_func(param.get_data(), param.get_grad())
            param.update(delta)