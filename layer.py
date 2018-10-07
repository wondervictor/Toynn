"""


"""

import numpy as np
import parameter


class FullyConnected:

    def __init__(self, mode,
                 in_feature,
                 out_feature,
                 weight_initializer=None,
                 bias_initializer=None):

        self.weights = None
        self.bias = None
        self.cache = None
        self.weights_shape = (in_feature, out_feature)
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        with_grad = False
        if mode == 'train':
            with_grad = True
        # initialize parameters
        self._initialize(with_grad=with_grad)

    def forward(self, x):
        # x.shape: [N, D1]
        # w.shape: [D1, D2]
        # b.shape: [D2]
        assert x.shape[1] == self.weights.shape[0]
        y = np.matmul(x, self.weights.get_data()) + self.bias.get_data()
        self.cache = x
        return y

    def _initialize(self, with_grad=True):
        if self.weight_initializer is None:
            self.weight_initializer = parameter.GaussianInitializer()
        if self.bias_initializer is None:
            self.bias_initializer = parameter.ConstantInitializer()

        self.weights = parameter.Parameter(shape=self.weights_shape,
                                           with_grad=with_grad,
                                           initializer=self.weight_initializer)

        self.bias = parameter.Parameter(shape=[self.weights_shape[-1]],
                                        with_grad=with_grad,
                                        initializer=self.bias_initializer)

    def backward(self, grad_in):
        # grad_in.shape: [N, D2]
        assert grad_in.shape[1] == self.weights_shape[1]
        # bias gradient
        self.bias.set_grad(np.sum(grad_in, 0))

        x = self.cache

        # weights gradient
        # (D1, N) * (N, D2) --> (D1, D2)
        grad_w = np.matmul(x.T, grad_in)
        self.weights.set_grad(grad_w)

        # x gradient
        # (N, D2) * (D2, D1) -> (N, D1)
        grad_out = np.matmul(grad_in, self.weights.get_data().T)
        return grad_out

    def __call__(self, x):
        return self.forward(x)


class BatchNorm:

    pass

