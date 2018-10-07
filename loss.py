"""

Loss Layer

"""

import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self._cache = None

    def forward(self, pred, target):
        # pred: [N,D]
        # target: [N,D]

        y = -np.sum(target * np.log(pred))
        self._cache = pred, target
        return y/pred.shape[0]

    def backward(self):
        pred, target = self._cache
        grad = - target / (pred*pred.shape[0])
        return grad

    def __call__(self, pred, target):
        return self.forward(pred, target)

    def __repr__(self):
        return "Loss: Cross Entropy Loss"


class SquareLoss:

    def __init__(self):
        self._cache = None

    def forward(self, pred, target):
        """ Square Loss (L2 Loss)
        Args:
            pred:
            target:
        Return:
        """
        assert pred.shape == target.shape, "pred.shape should be same with target.shape"
        self._cache = pred, target
        return np.mean(np.square((pred-target)))

    def backward(self):
        pred, target = self._cache
        grad = 2*(pred - target)
        return grad / (pred.shape[0]*pred.shape[1])

    def __call__(self, pred, target):

        return self.forward(pred, target)

    def __repr__(self):
        return "Loss: Square Loss"


class L1Loss:

    def __init__(self):
        self._cache = None

    def forward(self, pred, target):
        pass

    def backward(self):
        pass

    def __repr__(self):
        return "Loss: L1 Loss"
