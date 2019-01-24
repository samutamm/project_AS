import torch
from torch import nn
from copy import deepcopy

import numpy as np
from src.cnn_mnist import *

class CModule(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.c_mask = nn.Parameter(torch.ones(size))




class Masked_model(nn.Module):
    def __init__(self, model):
        super(Masked_model, self).__init__()

        original_modules = list(model.children())
        c_masks_mapping = {}
        c_masks_list = []
        i = 0
        for c in original_modules:
            for n, p in c.named_parameters():
                name = 'mask_' + n.replace('.', '_')
                c_masks_mapping[name] = i
                c_masks_list = nn.Parameter(torch.ones(p.size()))
                i += 1
                # self.register_parameter(name, c_masks[n])

        self.modules = nn.Sequential(* original_modules)
        self.c_masks = nn.ModuleList(c_masks_list)
        self.c_masks_mapping = c_masks_mapping

    def forward(self, x):

        for name, param in self.modules.named_parameters():
            mask_name = 'mask_' + name.replace('.', '_')
            m_param = self.c_masks[self.c_masks_mapping[mask_name]]
            param.data *= m_param.data

        output = self.modules(x)
        return output


class SNIP(nn.Module):
    def __init__(self, model):
        super(SNIP, self).__init__()
        self.model = model
        self.C_masks = nn.ParameterDict()
        self.mask_name = []
        self.init_C()

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

        # X, y = next(iter(data_loader))
        # print(X, y)
        # # Register C_var
        # for k in self.C_masks.keys():
        #     print(self.C_masks[k])
        # print(self.C_masks)
        # print(list(self.C_masks.keys()))
        # print(self.model.named_parameters())
        #
        # print(self.model.named_parameters()['mask_conv2_weight'])
        state_dict = self.model.state_dict()
        # for name, param in self.model.named_parameters():
        #     mask_name = 'mask_' + name.replace('.', '_')
        #     if name in self.parameters_name:
        #         for m_name, m_param in self.model.named_parameters():
        #             if m_name == mask_name:
        #                 # print(m_param)
        #                 param.data *= m_param.data
            # print(name, param)
        # print(self.model.state_dict())

        #
        # for i, param in enumerate(self.model.parameters()):
        #     print(i)
        #     param.data = nn.Parameter(param.data * self.C_masks[list(self.C_masks.keys())[i]])
        # #
        # # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        # # optimizer.zero_grad()

        self.masked_model = Masked_model(self.model)
        output = self.masked_model(X)
        loss = F.nll_loss(output, y)
        print(loss.item())
        loss.backward()
        # optimizer.step()


        mask = self.masked_model.c_masks
        for m in mask:
            print(mask[m].grad)
        # for name, param in self.model.named_parameters():
        #     if name in self.mask_name:
        #         print(param.grad)

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

    # device = torch.device("cuda" if use_cuda else "cpu")
    device = "cpu"
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
    print(sniped_model.get_total_param_number())
    # print(sniped_model.model.children())

    for c in sniped_model.model.children():
        for n, p in c.named_parameters():
            print(n, p)

    print(sniped_model.C_masks.children())
    for c in sniped_model.C_masks:
        print(c)
    # print(sniped_model.model.parameters()['mask_conv1_weight'])

    sniped_model.compute_mask(X, y, K=10)
    # C_masks = sniped_model.C_masks
    # for i, param in enumerate(model.parameters()):
    #     # print(param)
    #     param.data.copy_(param.data * C_masks[list(C_masks.keys())[i]])

    # C_test = deepcopy(model.state_dict())
