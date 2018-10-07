"""

"""
import tqdm
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

    initializer = parameter.GaussianInitializer(std=0.1)
    bias_initializer = parameter.ConstantInitializer(0.1)
    model = network.Network()
    model.add(
        FullyConnected(
            in_feature=784,
            out_feature=512,
            weight_initializer=initializer,
            bias_initializer=bias_initializer), name='fc1')
    model.add(ReLU(), name='leaky_relu1')
    model.add(
        FullyConnected(
            in_feature=512,
            out_feature=256,
            weight_initializer=initializer,
            bias_initializer=bias_initializer), name='fc2')
    model.add(ReLU(), name='leaky_relu2')
    model.add(
        FullyConnected(
            in_feature=256,
            out_feature=128,
            weight_initializer=initializer,
            bias_initializer=bias_initializer), name='fc23')
    model.add(ReLU(), name='leaky_relu23')
    model.add(
        FullyConnected(
            in_feature=128,
            out_feature=10,
            weight_initializer=initializer,
            bias_initializer=bias_initializer), name='fc4')
    model.add(Softmax(), name='softmax')

    model.add_loss(CrossEntropyLoss())
    lr = 0.8
    optimizer = SGD(lr=lr)

    print(model)
    for k, v in model.parameters().items():
        print(k, v)
    traingset = fetch_traingset()
    testset = fetch_testingset()

    test_images, test_labels = testset['images'], testset['labels']
    test_labels = np.array(test_labels)
    images, labels = traingset['images'], traingset['labels']
    batch_size = 512
    training_size = len(images)
    pbar = tqdm.tqdm(range(100))
    for epoch in pbar:
        losses = []
        for i in range(int(training_size/batch_size)):
            batch_images = np.array(images[i*batch_size:(i+1)*batch_size])
            batch_labels = np.array(labels[i*batch_size:(i+1)*batch_size])
            batch_labels = one_hot(batch_labels, 10)
            _, loss = model.forward(batch_images, batch_labels)
            losses.append(loss)
            model.backward()
            model.optimize()

        predicts = np.zeros((len(test_labels)))
        for i in range(int(len(test_labels)/1000)):
            batch_images = np.array(test_images[i*1000:(i+1)*1000])
            pred = model.forward(batch_images)
            pred = np.argmax(pred, 1)
            predicts[i*1000:(i+1)*1000] = pred

        acc = np.sum(test_labels == predicts) * 100 / len(test_labels)
        pbar.set_description('e:{} loss:{} acc:{}%'.format(epoch, float(np.mean(losses)), acc))


if __name__ == '__main__':

    mnist_model()







