"""

Loss Layer

"""

import numpy as np


class CrossEntropyLoss:

    cache = None

    def __init__(self):
        pass

    def forward(self, pred, target):
        pass

    def backward(self):
        pass


class SquareLoss:

    cache = None

    def __init__(self):
        pass

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
        return grad


class L1Loss:

    cache = None

    def __init__(self):
        pass

    def forward(self, pred, target):
        pass

    def backward(self):
        pass
