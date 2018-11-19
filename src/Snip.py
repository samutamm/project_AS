
import torch
from torch import nn

import numpy as np


def weights_init_uniform(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        #torch.nn.init.xavier_uniform(m.bias.data)

class SNIP:

    def __init__(self, model):
        self.model = model
        self.C = {}

        self.model.apply(weights_init_uniform) # algorithm line 1

        for i, param in enumerate(self.model.parameters()):
            if param.requires_grad:
                self.C[i] = torch.zeros(param.shape)

        self.S = self.C.copy()

    def prune(self, criterion, data_loader, K):
        """
        Algorithm 1 SNIP: Single-shot Network Pruning based on Connection Sensitivity
        from the paper with corresponding name. https://openreview.net/pdf/3b4408062d47079caf01147df0e4321eb792f507.pdf

        This function implements only lines 1-5 of the algorithm.

        :param loss_function: Loss function that is used later in learning to optimize given task.
        :param training_X: learning examples for calculating Connection Sensivity (line 3)
        :param training_y: labels for examples
        :param K: number of non-zero weights to be returned in output C
        :return: binary vector C where 1's are weights to be retained and 0's weights to drop
        """
        X, y = next(iter(data_loader))

        # maybe use the same optimiser than in normal learning process later
        optimizer = torch.optim.SGD(self.model.parameters(), 0.1, momentum=0.9)

        output = self.model(X)
        loss = criterion(output, y)
        loss.backward()

        for i, param in enumerate(self.model.parameters()):
            if param.requires_grad:
                self.S[i] = torch.abs(param.grad) / torch.sum(torch.abs(param.grad))
                s_values = self.S[i].numpy().flatten()
                np.sort(s_values)
                s_values = s_values[::-1]
                s_k = s_values[K]

                s_k_ = torch.ones(self.C[i].shape)
                s_k_ = s_k_.new_full(self.C[i].shape, float(s_k))

                self.C[i][self.S[i] > s_k_] = 1

        # TODO count Connection Sensivity
        # TODO sort s and take top-k
        return self.C
