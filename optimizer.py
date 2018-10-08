"""


"""

import numpy as np


class SGD:

    def __init__(self, lr):
        self.learning_rate = lr
        self.state = None

    def set_lr(self, lr):
        self.learning_rate = lr

    def optimize(self, grad, name):
        delta = grad * self.learning_rate
        return delta


class Momentum:

    learning_rate = 1e-2

    def __init__(self, lr, momentum=0.9):
        self.learning_rate = lr
        self.state = dict()
        self.momentum = momentum

    def set_lr(self, lr):
        self.learning_rate = lr

    def optimize(self, grad, name):
        v = self.state.get(name, np.zeros_like(grad))
        v = self.momentum * v - grad * self.learning_rate
        self.state[name] = v
        return -v


class Adam:

    learning_rate = 1e-2

    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self.learning_rate = lr
        self.state = dict()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8

    def set_lr(self, lr):
        self.learning_rate = lr

    def optimize(self, grad, name):
        m = self.state.get(name+':m', np.zeros_like(grad))
        v = self.state.get(name+':v', np.zeros_like(grad))
        t = self.state.get(name+':t', 1)

        t = t + 1
        self.state[name+':t'] = t

        beta1 = self.beta1
        beta2 = self.beta2
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_unbias = m / (1 - beta1 ** t)
        v_unbias = v / (1 - beta2 ** t)
        delta = self.learning_rate * m_unbias / (np.sqrt(v_unbias) + self.epsilon)

        self.state[name+':m'] = m
        self.state[name+':v'] = v

        return delta

