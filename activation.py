"""

"""

import numpy as np


class Sigmoid:

    def __init__(self, name='sigmoid'):
        self.name = name
        self._cache = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x.shape: [N, D]
        y = 1.0/(1+np.exp(-x))
        self._cache = y
        return y

    def backward(self, grad_in):
        y = self._cache
        return (1-y)*y*grad_in

    def __repr__(self):
        return "Actication: Sigmoid"


class ReLU:

    def __init__(self, name='relu'):
        self.name = name
        self._cache = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x.shape: [N, D]
        mask = x > 0
        self._cache = mask
        return mask * x

    def backward(self, grad_in):
        # grad_in.shape: [N, D]
        mask = self._cache
        return mask * grad_in

    def __repr__(self):
        return "Actication: ReLU"


class LeakyReLU:

    def __init__(self, name='leaky_relu', negative_slope=0.2):
        self.name = name
        self.negative_slope = negative_slope
        self._cache = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        mask = x > 0
        self._cache = mask
        return x*mask + (1-mask)*x*self.negative_slope

    def backward(self, grad_in):
        mask = self._cache
        return grad_in * mask + (1-mask)*grad_in*self.negative_slope

    def __repr__(self):
        return "Actication: LeakyReLU ({})".format(self.negative_slope)


class Softmax:

    def __init__(self, name='softmax'):
        self.name = name
        self._cache = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x.shape: [N, D]
        x_exp = np.exp(x)
        y = x_exp / np.sum(x_exp, axis=1, keepdims=True)
        self._cache = y
        return y

    def backward(self, grad_in):
        # grad_in.shape: [N,D]
        # y.shape: [N, D]
        y = self._cache
        ygrad = np.sum(y*grad_in, axis=1, keepdims=True)
        grad_out = y * (grad_in - ygrad)
        return grad_out

    def __repr__(self):
        return "Actication: Softmax"


class Tanh:

    def __init__(self, name='tanh'):
        self.name = name
        self._cache = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self._cache = x
        y = np.tanh(x)
        self._cache = y
        return y

    def backward(self, grad_in):
        y = self._cache
        grad_out = grad_in * (1-y**2)
        return grad_out

    def __repr__(self):
        return "Actication: Tanh"


class ELU:

    def __init__(self, name='elu', alpha=0.0):
        self.name = name
        self._cache = None
        self.alpha = alpha

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        mask = x >= 0
        y = mask * x + (1 - mask) * self.alpha * (np.exp(x)-1)
        self._cache = mask, y
        return x

    def backward(self, grad_in):
        mask, y = self._cache
        grad_out = grad_in * mask + (1-mask)*(y+self.alpha)
        return grad_out

    def __repr__(self):
        return "Actication: ELU"
