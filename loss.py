"""

Loss Layer

"""

import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self.cache = None

    def forward(self, pred, target):
        pass

    def backward(self):
        pass


class SquareLoss:

    def __init__(self):
        self.cache = None

    def forward(self, pred, target):
        """ Square Loss (L2 Loss)
        Args:
            pred:
            target:
        Return:
        """
        assert pred.shape == target.shape, "pred.shape should be same with target.shape"
        self.cache = pred, target
        return np.mean(np.square((pred-target)))

    def backward(self):
        pred, target = self.cache
        grad = 2*(pred - target)
        return grad / (pred.shape[0]*pred.shape[1])

    def __call__(self, pred, target):

        return self.forward(pred, target)


class L1Loss:

    def __init__(self):
        self.cache = None

    def forward(self, pred, target):
        pass

    def backward(self):
        pass
