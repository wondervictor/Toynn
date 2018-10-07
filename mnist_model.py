"""

"""

import random
import numpy as np
from activation import Softmax, ReLU, LeakyReLU
from layer import FullyConnected
from loss import CrossEntropyLoss
from optimizer import SGD
import parameter
import network
from utils import one_hot
from data.mnist import fetch_testingset, fetch_traingset


def mnist_model():

    model = network.Network()
    model.add(FullyConnected(in_feature=784, out_feature=512), name='fc1')
    model.add(LeakyReLU(), name='leaky_relu1')
    model.add(FullyConnected(in_feature=512, out_feature=256), name='fc2')
    model.add(LeakyReLU(), name='leaky_relu2')
    model.add(FullyConnected(in_feature=256, out_feature=256), name='fc3')
    model.add(LeakyReLU(), name='leaky_relu3')
    model.add(FullyConnected(in_feature=256, out_feature=128), name='fc4')
    model.add(LeakyReLU(), name='leaky_relu4')
    model.add(FullyConnected(in_feature=128, out_feature=10), name='fc5')
    model.add(Softmax(), name='softmax')

    model.add_loss(CrossEntropyLoss())

    optimizer = SGD(lr=1e-4)

    print(model)
    traingset = fetch_traingset()
    images, labels = traingset['images'], traingset['labels']
    batch_size = 256
    training_size = len(images)
    for epoch in range(50):
        for i in range(int(training_size/batch_size)):
            batch_images = np.array(images[i*batch_size:(i+1)*batch_size])
            batch_labels = np.array(labels[i*batch_size:(i+1)*batch_size])
            batch_labels = one_hot(batch_labels, 10)

            _, loss = model.forward(batch_images, batch_labels)
            print("e:{}, i:{} loss: {}".format(epoch, i, loss))
            model.backward()
            model.optimize(optimizer)


if __name__ == '__main__':

    mnist_model()







