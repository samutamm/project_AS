
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

    def calculate_mask(self, criterion, data_loader, K):
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
        #optimizer = torch.optim.SGD(self.model.parameters(), 0.1, momentum=0.9)

        output = self.model(X)
        loss = criterion(output, y)
        loss.backward()

        g, gradient_mapping = self.get_all_gradients()
        self.S = np.abs(g) / np.sum(np.abs(g))
        order = np.argsort(self.S)
        order = order[::-1]
        threshold = self.S[order[K]]

        C = np.ones(g.shape[0])
        C[self.S <= threshold] = 0

        self.C = C
        self.weight_mapping = gradient_mapping
        return C, gradient_mapping

    def get_all_gradients(self):
        params_id_mapping = {}
        params = []
        last_index = 0
        for i, param in enumerate(self.model.parameters()):
            if param.requires_grad:
                dimensions = list(param.grad.shape)
                params_vector = param.grad.data.numpy().flatten()
                param_indexes = np.arange(params_vector.shape[0])
                params.append(params_vector)
                for local_idx, _ in enumerate(param_indexes):
                    current_idx = last_index + local_idx
                    local_index_in_layer_i = np.unravel_index(local_idx, dimensions)
                    params_id_mapping[current_idx] = (i, local_index_in_layer_i)

                last_index += params_vector.shape[0]
        return np.concatenate(params), params_id_mapping

    def prune_parameters(self):
        """
        This method uses precalculated mask (the variable C in Algorithm 1) to set some params to zero,
        also known as pruning the model.
        :return:
        """
        assert len(self.C) > 0, "Please call calculate_mask before this function."
        state_dict = self.model.state_dict()
        layers = list(state_dict.keys())
        for i,c_value in enumerate(self.C):
            (layer_i, idx_in_layer) = self.weight_mapping[i]
            layer = state_dict[layers[layer_i]]
            layer[idx_in_layer].data *= c_value
        return state_dict
