"""

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data.mnist import fetch_testingset, fetch_traingset


def weight_init(m):

    if isinstance(m, nn.Linear):
        m.weight.data.normal_(std=1e-2)
        m.bias.data.fill_(0)


def mnist_model():

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    model.apply(weight_init)
    optimizer = optim.SGD(lr=1e-2, params=model.parameters())
    loss_criterion = nn.CrossEntropyLoss()
    traingset = fetch_traingset()
    images, labels = traingset['images'], traingset['labels']
    batch_size = 512
    training_size = len(images)
    for epoch in range(50):
        losses = []
        for i in range(int(training_size/batch_size)):
            batch_images = torch.Tensor(np.array(images[i*batch_size:(i+1)*batch_size]))
            batch_labels = torch.LongTensor(np.array(labels[i*batch_size:(i+1)*batch_size]).astype('int32'))
            y = model(batch_images)
            loss = loss_criterion(y, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print("e:{}, loss: {}".format(epoch, np.mean(losses)))


if __name__ == '__main__':

    mnist_model()
