import os
import sys

import torch
from torch.optim import SGD
from torch import nn

## Access src directory from ./notebooks/ folder
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

from src.Snip_copy import SNIP
from src.models.AlexNet_s import AlexNet_s
from src.optimization import epoch

import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import pickle as pkl

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_data_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

def train_model(model, snip = None, epochs = 20):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)

    model = model.cuda()
    criterion = criterion.cuda()
    
    if snip:
        hooks = snip.register_masks()
        assert snip.K == snip.get_nonzero_param_number()

    train_losses = []
    test_losses = []
    accuracys = []
    # On itère sur les epochs
    for i in range(epochs):
        print("=================\n=== EPOCH "+str(i+1)+" =====\n=================\n")
        # Phase de train
        _, loss = epoch(train_data_loader, model, criterion,snip_pruning=snip, optimizer=optimizer)
        # Phase d'evaluation
        with torch.no_grad():
            acc_test, loss_test = epoch(test_data_loader, model, criterion)

        train_losses.append(loss.avg)
        test_losses.append(loss_test.avg)
        accuracys.append(acc_test.avg)

    if snip:
        for hook in hooks:
            hook.remove()
        nonzero_params = snip.get_nonzero_param_number()
        print(nonzero_params)
        assert snip.K == nonzero_params

    return train_losses, test_losses, accuracys

def print_losses_and_acc(training_losses_, test_losses_, acc_):
    plt.title("L'erreur moyenne d'un batch")
    plt.plot(np.arange(len(training_losses_)), training_losses_, label="Train")
    plt.plot(np.arange(len(test_losses_)), test_losses_, label="Test")
    plt.ylabel("L'erreur")
    plt.xlabel('Epochs')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(len(acc_)), acc_)
    plt.show()


# import time
# for i in range(5):
#     original_model = AlexNet_s()
#     train_losses, test_losses, accuracys  = train_model(original_model, epochs = 50)
#     pkl.dump([train_losses, test_losses, accuracys], open('org-alexnet-cifar-'+str(i)+'.pkl', 'wb'))
#     time.sleep(60)
for ratio in [1,5,10,20]:
    prune_model = AlexNet_s()
    snip = SNIP(prune_model)
    total_param_number = snip.get_total_param_number()
    print("Original number of params : {}".format(total_param_number))
    K = total_param_number // ratio
    print("nb of params : {}".format(K))

# import time
# for ratio in [10,20]:
# 	for i in range(5):
# 		prune_model = AlexNet_s()
# 		snip = SNIP(prune_model)
# 		total_param_number = snip.get_total_param_number()
# 		print("Original number of params : {}".format(total_param_number))
# 		K = total_param_number // ratio
# 		print("10% of params : {}".format(K))
# 		C_masks = snip.compute_mask(train_data_loader, K=K)

# 		train_losses, test_losses, accuracys  = train_model(prune_model, snip, epochs = 50)
# 		pkl.dump([train_losses, test_losses, accuracys], open(str(ratio)+'-alexnet-cifar-'+str(i)+'.pkl', 'wb'))
# 		time.sleep(60)