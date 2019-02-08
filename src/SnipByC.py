import torch
from torch import nn
from copy import deepcopy

import numpy as np

import torch.nn.functional as F

class CLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        w = torch.FloatTensor(output_size, input_size)
        b = torch.FloatTensor(output_size)
        torch.nn.init.xavier_uniform_(w.data)
        torch.nn.init.uniform_(b)
        #torch.nn.init.xavier_uniform_(b)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)
        self.c_mask = nn.Parameter(torch.ones((output_size, input_size)))

#        torch.nn.init.xavier_uniform_(self.b.data) no one dimensional initialisation

    def forward(self, x):
        masked_w = self.w * self.c_mask
        activation = F.linear(x, masked_w, self.b)
        return activation


class Masked_model(nn.Module):
    def __init__(self):
        super(Masked_model, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5), stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Conv2d(6, 16, (5, 5), stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            CLinear(400, 120),
            nn.Tanh(),
            CLinear(120, 84),
            nn.Tanh(),
            CLinear(84, 10)
            # Rappel : Le softmax est inclus dans la loss, ne pas le mettre ici
        )

    def forward(self, x):
        bsize = x.size(0)  # taille du batch
        output = self.features(x)  # on calcule la sortie des conv
        output = output.view(bsize, -1)  # on applati les feature map 2D en un
        # vecteur 1D pour chaque input
        output = self.classifier(output)  # on calcule la sortie des fc
        return output


class SNIP:
    def __init__(self):
        self.masked_model = Masked_model()

    def get_total_param_number(self):
        return sum([
            np.prod(list(param.data.shape))
            for (name, param) in self.masked_model.classifier.named_parameters()
            if param.requires_grad and 'w' in name])

    def get_nonzero_param_number(self):
        return sum([
            torch.nonzero(param.data).shape[0]
            for (name, param) in self.masked_model.classifier.named_parameters()
            if param.requires_grad and 'w' in name])

    def compute_mask(self, X, y, K):
        """
        Algorithm 1 SNIP: Single-shot Network Pruning based on Connection Sensitivity
        from the paper with corresponding name. https://openreview.net/pdf/3b4408062d47079caf01147df0e4321eb792f507.pdf

        This function implements only lines 1-5 of the algorithm.

        :param K: number of non-zero weights to be garded in model
        """
        self.K = K

        criterion = nn.CrossEntropyLoss()

        output = self.masked_model(X)
        loss = criterion(output, y)
        print("loss : ", loss.item())
        loss.backward()
        # optimizer.step()

        g, gradient_mapping = self.get_all_gradients()
        self.S = np.abs(g) / np.sum(np.abs(g))
        order = np.argsort(self.S)
        order = order[::-1]
        threshold = self.S[order[K]]

        if np.isnan(g).sum() > 0:
            print(np.isnan(g).sum())
            import pdb; pdb.set_trace()

        C = np.ones(g.shape[0])
        C[self.S <= threshold] = 0

        with torch.no_grad():
            self.add_C_values_to_model(C, gradient_mapping)

        return self.masked_model


    def get_all_gradients(self):
        """
        Iterate all layers and get the gradients of each layer.
        :return:
        """
        params_id_mapping = {}
        params = []
        last_index = 0
        i = 0
        for name, param in self.masked_model.classifier.named_parameters():
            if 'w' not in name:
                continue
            if param.requires_grad:
                dimensions = list(param.grad.shape)
                params_vector = param.grad.data.numpy().flatten()
                param_indexes = np.arange(params_vector.shape[0])
                params.append(params_vector)
                for local_idx, _ in enumerate(param_indexes):
                    current_idx = last_index + local_idx
                    local_index_in_layer_i = np.unravel_index(local_idx, dimensions)
                    if len(dimensions) == 1:
                        import pdb; pdb.set_trace()
                    params_id_mapping[current_idx] = (i, local_index_in_layer_i)

                last_index += params_vector.shape[0]
                i+=1

        self.masked_model.zero_grad()
        return np.concatenate(params), params_id_mapping


    def add_C_values_to_model(self, C, weight_mapping):
        """
        :param All values of C in one long vector:
        """

        layer_index_mapping = {}
        i = 0
        for j, layer in enumerate(self.masked_model.classifier):
            if layer.__class__.__name__ != "CLinear":
                continue

            layer_index_mapping[i] = j
            shape = self.masked_model.classifier[j].c_mask.shape
            self.masked_model.classifier[j].c_mask = nn.Parameter(torch.zeros(shape).cuda()) # reinitialize
            self.masked_model.classifier[j].c_mask.requires_grad = False # freeze for training
            i += 1

        for i, c_value in enumerate(C):
            (layer_i, idx_in_layer) = weight_mapping[i]
            self.masked_model.classifier[layer_index_mapping[layer_i]].c_mask[idx_in_layer] = c_value


if __name__ == '__main__':
    use_cuda = True if torch.cuda.is_available() else False

    batch_size = 258
    test_batch_size = 1000
    lr = 0.01
    momentum = 0.5
    seed = 1
    epochs = 10
    #torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    #device = "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    X, y = next(iter(train_loader))

    sniped_model = SNIP()

    sniped_model.compute_mask(X, y, K=10)
