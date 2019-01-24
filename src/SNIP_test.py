import torch
from torch import nn
from copy import deepcopy

import numpy as np
from cnn_mnist import *

import torch.nn.functional as F

class CLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        w = torch.FloatTensor(output_size, input_size)
        b = torch.FloatTensor(output_size)
        torch.nn.init.xavier_uniform_(w)
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
    def __init__(self, model_orig):
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
    def __init__(self, model):
        #super(SNIP, self).__init__()
        self.model = model
        #self.C_masks = nn.ParameterDict()
        #self.mask_name = []
        #self.init_C()

    def get_total_param_number(self):
        return sum([np.prod(list(param.data.shape)) for param in self.model.parameters() if param.requires_grad])

    def get_nonzero_param_number(self):
        return sum([torch.nonzero(param.data).shape[0] for param in self.model.parameters() if param.requires_grad])

    def init_C(self):
        C_test = deepcopy(model.state_dict())
        self.parameters_name = list(C_test.keys())
        for k in C_test.keys():
            name = 'mask_' + k.replace('.', '_')
            # print(C_test[k].size())
            self.C_masks[name] = nn.Parameter(torch.ones(C_test[k].size()))
            # self.C_masks[k].requires_grad_()
            self.mask_name.append(name)
            self.model.register_parameter(name, self.C_masks[name])

    def compute_mask(self, X, y, K):
        """
        Algorithm 1 SNIP: Single-shot Network Pruning based on Connection Sensitivity
        from the paper with corresponding name. https://openreview.net/pdf/3b4408062d47079caf01147df0e4321eb792f507.pdf

        This function implements only lines 1-5 of the algorithm.

        :param K: number of non-zero weights to be garded in model
        :return: binary vector C where 1's are weights to be retained and 0's weights to drop
        """
        self.K = K

        criterion = nn.CrossEntropyLoss()

        self.masked_model = Masked_model(self.model)
        output = self.masked_model(X)
        loss = criterion(output, y)
        print(loss.item())
        loss.backward()
        # optimizer.step()

        for name, m in self.masked_model.classifier[0].named_parameters():
            print(name)
            print(m.grad)
            print(m.grad.sum())
            #if name == 'c_mask':
            #    import pdb; pdb.set_trace()
        # for name, param in self.model.named_parameters():
        #     if name in self.mask_name:
        #        print(param.grad)

        # for k in self.C_masks.keys():
        #     print(self.C_masks[k].grad)
        #
        # # for param in self.model.parameters():
        # #     print(param.grad.data)

    # def forward(self, input):
    #


if __name__ == '__main__':
    use_cuda = True if torch.cuda.is_available() else False

    batch_size = 64
    test_batch_size = 1000
    lr = 0.01
    momentum = 0.5
    seed = 1
    epochs = 10
    torch.manual_seed(seed)

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

    model = Net()
    sniped_model = SNIP(model)
    '''print(sniped_model.get_total_param_number())
    # print(sniped_model.model.children())

    for c in sniped_model.model.children():
        for n, p in c.named_parameters():
            print(n, p)

    print(sniped_model.C_masks.children())
    for c in sniped_model.C_masks:
        print(c)
    # print(sniped_model.model.parameters()['mask_conv1_weight'])
    '''
    sniped_model.compute_mask(X, y, K=10)
    # C_masks = sniped_model.C_masks
    # for i, param in enumerate(model.parameters()):
    #     # print(param)
    #     param.data.copy_(param.data * C_masks[list(C_masks.keys())[i]])

    # C_test = deepcopy(model.state_dict())

