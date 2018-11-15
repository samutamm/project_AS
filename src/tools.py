
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import Sigmoid, Linear, ReLU
from torch.nn.modules import Sequential
from torch.nn.modules.loss import CrossEntropyLoss

import numpy as np


def get_train_test_loaders(batch_size=64, label1 = 1, label2 = 3, binary=True):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                              transform=transforms.Compose(
                                  [transforms.ToTensor(),
                                  transforms.Normalize((0.1307,),(0.3081,))]
                              )),
        batch_size=batch_size,
        shuffle=True
    )

    if binary:
        labels = loader.dataset.train_labels
        mask = (labels == label1) + (labels == label2) > 0

        loader.dataset.train_data = loader.dataset.train_data[mask]
        loader.dataset.train_labels = loader.dataset.train_labels[mask]

        labels = torch.where(loader.dataset.train_labels == label1, torch.ones(1), -torch.ones(1))
        loader.dataset.train_labels = labels

    full_dataset = loader.dataset.train_data
    N = full_dataset.size()[0]
    train_size = int(0.7 * N)
    val_size = int(0.2 * N)
    test_size = N - train_size - val_size
    train_indices, val_indices, test_indices = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices.indices)
    )

    val_loader = torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices.indices)
    )

    test_loader = torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices.indices)
    )

    return train_loader, val_loader, test_loader, train_size, val_size, test_size


def get_minibatches(loader, device):
    for data, target in loader:
        normalize = torch.nn.BatchNorm2d(1)
        data = normalize(data)

        data = torch.squeeze(data)
        target = target.cuda(async=True)
        data = data.cuda(async=True)

        batch_n = data.size()[0]
        X = data.view(batch_n, -1)
        ones = torch.ones((X.size()[0], 1), device=device)
        X = torch.cat((X, ones), 1)

        y_onehot = torch.zeros((target.size()[0], 10), device=device)
        y_onehot.zero_()
        y_onehot.scatter_(1, target.view(-1, 1), 1)

        X = torch.autograd.Variable(X)
        y = torch.autograd.Variable(y_onehot)
        yield X, y.long()


def n_layer_nn(optimiser_function, layer_dims=[28 * 28 + 1, 128, 10], learning_rate=0.1, epochs=100):
    train_loader, val_loader, test_loader, train_size, val_size, test_size = get_train_test_loaders(binary=False)
    torch.cuda.set_device(0)
    device = torch.device('cuda')

    layers = len(layer_dims)
    assert layers >= 3, "Please give at leaset 3 dimensions"

    modules = [Linear(layer_dims[0], layer_dims[1]), ReLU()]
    for i in range(1, layers - 2):
        modules.append(Linear(layer_dims[i], layer_dims[i + 1]))
        modules.append(ReLU())

    modules.append(Linear(layer_dims[layers - 2], layer_dims[layers - 1]))
    modules.append(Sigmoid())
    print(modules)
    model = Sequential(*modules).cuda('cuda:0')

    loss_function = CrossEntropyLoss()

    optimiser = optimiser_function(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    accuracy = []

    for epoch in range(epochs):
        losses = []
        for i, (X, y) in enumerate(get_minibatches(train_loader, device)):
            optimiser.zero_grad()
            yhat = model.forward(X)
            loss = loss_function(yhat, y.argmax(1))
            losses.append(loss.item())
            loss.backward()
            optimiser.step()
            import pdb; pdb.set_trace()

        train_losses.append(np.mean(losses))

        if epoch % 3 == 0:
            with torch.no_grad():
                losses = []
                corrects = 0
                for i, (X, y) in enumerate(get_minibatches(val_loader, device)):
                    y = y.argmax(1)
                    yhat = model.forward(X)
                    losses.append(loss_function(yhat, y).item())
                    ypred = yhat.argmax(1)
                    corrects += (ypred == y).sum()
                val_loss = np.mean(losses)
                val_losses.append(val_loss)
                acc = corrects.cpu().numpy() / val_size
                # print("Accuracy {}".format(acc))
                accuracy.append(acc)
    return val_losses, accuracy
