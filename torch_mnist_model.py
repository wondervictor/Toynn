"""

"""
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data.mnist import fetch_testingset, fetch_traingset


def weight_init(m):

    if isinstance(m, nn.Linear):
        m.weight.data.normal_(std=1e-1)
        m.bias.data.normal_(std=1e-1)


def mnist_model():

    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    model.apply(weight_init)
    optimizer = optim.Adam(lr=1e-3, params=model.parameters())
    loss_criterion = nn.CrossEntropyLoss()
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
            batch_images = torch.Tensor(np.array(images[i*batch_size:(i+1)*batch_size]))
            batch_labels = torch.LongTensor(np.array(labels[i*batch_size:(i+1)*batch_size]).astype('int32'))
            y = model(batch_images)
            loss = loss_criterion(y, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        predicts = np.zeros((len(test_labels)))
        for i in range(int(len(test_labels)/1000)):
            batch_images = torch.Tensor(np.array(test_images[i*1000:(i+1)*1000]))
            pred = model(batch_images).data.numpy()
            pred = np.argmax(pred, 1)
            predicts[i*1000:(i+1)*1000] = pred

        acc = np.sum(test_labels == predicts) * 100 / len(test_labels)
        pbar.set_description('e:{} loss:{} acc:{}%'.format(epoch, float(np.mean(losses)), acc))


if __name__ == '__main__':

    mnist_model()
