import torch
from torch import nn
from copy import deepcopy

import numpy as np

def weights_init_uniform(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        # torch.nn.init.xavier_uniform(m.bias.data)


class SNIP:

    def __init__(self, model):
        """
        :param model:
        """
        self.model = model.cuda()
        self.C_masks = {}

        self.model.apply(weights_init_uniform)  # algorithm line 1

    def get_total_param_number(self):
        return sum([np.prod(list(param.data.shape)) for param in self.model.parameters() if param.requires_grad])

    def get_nonzero_param_number(self):
        return sum([torch.nonzero(param.data).shape[0] for param in self.model.parameters() if param.requires_grad])

    def reshape_mask_layer_by_layer(self, C, weight_mapping):
        """
        :param All values of C in one long vector:
        :return: One C mask for each layer.
        """
        state_dict = self.model.state_dict()
        layers = list(state_dict.keys())

        layers_masks = {}
        for layer in layers:
            layers_masks[layer] = torch.zeros(state_dict[layer].shape).cuda()

        for i, c_value in enumerate(C):
            (layer_i, idx_in_layer) = weight_mapping[i]
            layers_masks[layers[layer_i]][idx_in_layer] = c_value

        return layers_masks

    def compute_mask(self, data_loader, K):
        """
        Algorithm 1 SNIP: Single-shot Network Pruning based on Connection Sensitivity
        from the paper with corresponding name. https://openreview.net/pdf/3b4408062d47079caf01147df0e4321eb792f507.pdf

        This function implements only lines 1-5 of the algorithm.

        :param K: number of non-zero weights to be garded in model
        :return: binary vector C where 1's are weights to be retained and 0's weights to drop
        """
        self.K = K

        X, y = next(iter(data_loader))

        # RNN-MNIST
        # sequence_length = 28
        # input_size = 28
        # X = X.reshape(-1, sequence_length, input_size).cuda()
        criterion = nn.CrossEntropyLoss()

        # maybe use the same optimiser than in normal learning process later
        output = self.model(X.cuda())
        loss = criterion(output, y.cuda())
        loss.backward()

        g, gradient_mapping = self.get_all_gradients()
        self.S = np.abs(g) / np.sum(np.abs(g))
        order = np.argsort(self.S)
        order = order[::-1]
        threshold = self.S[order[K]]

        C = np.ones(g.shape[0])
        C[self.S <= threshold] = 0

        self.C_masks = self.reshape_mask_layer_by_layer(C, gradient_mapping)
        return self.C_masks

    # def compute_mask(self, X, y, K):
    #     """
    #     Algorithm 1 SNIP: Single-shot Network Pruning based on Connection Sensitivity
    #     from the paper with corresponding name. https://openreview.net/pdf/3b4408062d47079caf01147df0e4321eb792f507.pdf
    #
    #     This function implements only lines 1-5 of the algorithm.
    #
    #     :param K: number of non-zero weights to be garded in model
    #     :return: binary vector C where 1's are weights to be retained and 0's weights to drop
    #     """
    #     self.K = K
    #
    #     output = self.model(X)
    #     loss = F.nll_loss(output, y)
    #     print(loss.item())
    #     loss.backward()
    #
    #     # X, y = next(iter(data_loader))
    #     #
    #     # criterion = nn.CrossEntropyLoss()
    #     #
    #     # # maybe use the same optimiser than in normal learning process later
    #     # output = self.model(X)
    #     # loss = criterion(output, y)
    #     # loss.backward()
    #
    #     g, gradient_mapping = self.get_all_gradients()
    #     self.S = np.abs(g) / np.sum(np.abs(g))
    #     order = np.argsort(self.S)
    #     order = order[::-1]
    #     threshold = self.S[order[K]]
    #
    #     C = np.ones(g.shape[0])
    #     C[self.S <= threshold] = 0
    #
    #     self.C_masks = self.reshape_mask_layer_by_layer(C, gradient_mapping)
    #     return self.C_masks

    def get_all_gradients(self):
        """
        Iterate all layers and get the gradients of each layer.
        :return:
        """
        params_id_mapping = {}
        params = []
        last_index = 0
        for i, param in enumerate(self.model.parameters()):
            if param.requires_grad:
                dimensions = list(param.grad.shape)
                params_vector = param.data.cpu().numpy().flatten()
                params_grad_vector = param.grad.data.cpu().numpy().flatten()
                param_indexes = np.arange(params_vector.shape[0])
                params.append(np.multiply(params_vector, params_grad_vector))
                for local_idx, _ in enumerate(param_indexes):
                    current_idx = last_index + local_idx
                    local_index_in_layer_i = np.unravel_index(local_idx, dimensions)
                    params_id_mapping[current_idx] = (i, local_index_in_layer_i)

                last_index += params_vector.shape[0]
        self.model.zero_grad()
        return np.concatenate(params), params_id_mapping

    def register_masks(self):
        """
        Registers masks to be computed after gradient.
        :return:
        """
        assert len(self.C_masks) > 0, "Please call compute_mask before this function."
        state_dict = self.model.state_dict()
        param_names = list(state_dict.keys())
        hooks = []
        for i, (param_name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad and param_name in param_names:
                assert param.data.shape == self.C_masks[param_name].shape

                # Let's apply mask for initialized weights
                param.data *= self.C_masks[param_name]

                # Let's register hook that is applied after gradient is computed
                # --> this way zero weights stays zero
                hook = param.register_hook(lambda grad, mask=self.C_masks[param_name]: grad * mask)
                hooks.append(hook)

        return hooks


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

# if __name__ == '__main__':
#     use_cuda = True if torch.cuda.is_available() else False
#
#     batch_size = 64
#     test_batch_size = 1000
#     lr = 0.01
#     momentum = 0.5
#     seed = 1
#     epochs = 10
#     torch.manual_seed(seed)
#
#     # device = torch.device("cuda" if use_cuda else "cpu")
#     device = torch.device("cpu")
#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#         batch_size=test_batch_size, shuffle=True, **kwargs)
#
#     X, y = next(iter(train_loader))
#
#     model = Net()
#     sniped_model = SNIP(model)
#     total_param_number = sniped_model.get_total_param_number()
#     K = total_param_number // 10
#
#     sniped_model.compute_mask(X, y, K=10)
#
#     train_losses, test_losses, accuracys = train(prune_model, snip)
