"""
Network
"""


class Network:

    def __init__(self):
        self._params = dict()
        self.layers = []
        self.loss_layer = None

    def add(self, layer, name):
        assert name is not None
        assert layer is not None

        self.layers.append(layer)
        if layer.params is not None:
            for k, v in layer.params.items():
                self._params[name+':'+k] = v

    def add_loss(self, loss_layer):
        self.loss_layer = loss_layer

    def forward(self, x, y=None):
        for layer in self.layers:
            x = layer(x)

        if y is not None and self.loss_layer is not None:
            # compute loss
            loss = self.loss_layer(x, y)
            return x, loss

        return x

    def backward(self):
        num_layers = len(self.layers)
        grad = self.loss_layer.backward()
        for i in range(num_layers-1, 0, -1):
            grad = self.layers[i].backward(grad)

        # input's grad
        return grad

    def optimize(self, optimizer):
        optimizer.optimize(self._params)

    def save_params(self):
        # TODO: save parameters to file
        pass

    def load_params(self):
        # TODO: load parameters from file
        pass

    def __repr__(self):

        for i in range(len(self.layers)):
            print("[{}]: {}".format(i, type(self.layers[i])))

    def __call__(self, x, y=None):
        return self.forward(x, y)

