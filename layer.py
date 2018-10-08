"""


"""

import numpy as np
import parameter


class FullyConnected:

    def __init__(self,
                 name,
                 in_feature,
                 out_feature,
                 reg=1e-4,
                 mode='train',
                 weight_initializer=None,
                 bias_initializer=None):

        self.name = name
        self.weights = None
        self.bias = None
        self._cache = None
        self._reg = reg
        self._weights_shape = (in_feature, out_feature)
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.params = dict()

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
        self._cache = x
        return y

    def _initialize(self, with_grad=True):
        if self.weight_initializer is None:
            self.weight_initializer = parameter.GaussianInitializer()
        if self.bias_initializer is None:
            self.bias_initializer = parameter.ConstantInitializer()

        self.weights = parameter.Parameter(shape=self._weights_shape,
                                           with_grad=with_grad,
                                           initializer=self.weight_initializer)

        self.bias = parameter.Parameter(shape=[self._weights_shape[-1]],
                                        with_grad=with_grad,
                                        initializer=self.bias_initializer)

        self.params['weights'] = self.weights
        self.params['bias'] = self.bias

    def backward(self, grad_in):
        # grad_in.shape: [N, D2]
        assert grad_in.shape[1] == self._weights_shape[1]
        # bias gradient
        self.bias.set_grad(np.sum(grad_in, 0))

        x = self._cache

        # weights gradient
        # (D1, N) * (N, D2) --> (D1, D2)
        grad_w = np.matmul(x.T, grad_in) + self._reg * self.weights.get_data()
        self.weights.set_grad(grad_w)

        # x gradient
        # (N, D2) * (D2, D1) -> (N, D1)
        grad_out = np.matmul(grad_in, self.weights.get_data().T)
        return grad_out

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "FullyConnected ({},{})".format(self._weights_shape[0], self._weights_shape[1])


class BatchNorm:

    def __init__(self, name, num_features, mode='train', momentum=0.9, eps=1e-5):
        self.name = name
        self.gamma = None
        self.beta = None
        self._cache = None
        self.params = dict()
        self.running_mean = None
        self.running_var = None
        self.momentum = momentum
        self.eps = eps
        self.mode = mode
        self.num_features = num_features
        self._initialize()

    def _initialize(self):
        self.gamma = parameter.Parameter(shape=[self.num_features], initializer=parameter.ConstantInitializer(1.0))
        self.beta = parameter.Parameter(shape=[self.num_features], initializer=parameter.ConstantInitializer(0.0))
        self.running_mean = np.zeros([self.num_features], dtype='float32')
        self.running_var = np.zeros([self.num_features], dtype='float32')
        self.params['gamma'] = self.gamma
        self.params['beta'] = self.beta

    def forward(self, x):
        # x.shape: [N, D]
        if self.mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.sum(np.square((x - sample_mean)), axis=0) / x.shape[0]
            self.running_mean = sample_mean * (1 - self.momentum) + self.running_mean * self.momentum
            self.running_var = sample_var * (1 - self.momentum) + self.running_var * self.momentum
            std_var = (np.sqrt(sample_var + self.eps))
            x_ = (x - sample_mean) / std_var
            out = self.gamma.get_data() * x_ + self.beta.get_data()
            self._cache = x, x_, sample_mean, std_var, sample_var
        else:
            x_ = (x - self.running_mean) / (np.sqrt(self.running_var + self.eps))
            out = self.gamma.get_data() * x_ + self.beta.get_data()

        return out

    def backward(self, grad_in):
        x, x_, sample_mean, sqrt_var, var = self._cache
        N, D = grad_in.shape
        dx = grad_in * self.gamma.get_data()

        dbeta = np.sum(grad_in, axis=0)
        dgamma = x_ * grad_in
        dgamma = np.sum(dgamma, 0)

        dx = (1. / N) * 1 / sqrt_var * (N * dx - np.sum(dx, axis=0) - x_ * np.sum(dx * x_, axis=0))

        self.gamma.set_grad(dgamma)
        self.beta.set_grad(dbeta)

        return dx

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):

        return "Batch Norm"


class Dropout:

    def __init__(self, name, mode='train', droprate=0.5):
        self.name = name
        self._cache = None
        self.dropout_rate = droprate
        self.mode = mode

    def forward(self, x):

        if self.mode == 'train':
            probs = np.random.rand(*x.shape)
            mask = probs > self.dropout_rate
            out = mask * x
            self._cache = mask
        else:
            out = x

        return out

    def backward(self, grad_in):
        mask = self._cache
        return mask * grad_in

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Dropout({})".format(self.dropout_rate)

