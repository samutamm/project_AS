import os
import sys

import torch
from torch.optim import SGD
from torch import nn

## Access src directory from ./notebooks/ folder
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

from src.Snip_copy import SNIP
# from src.models import ConvNet
from src.optimization import epoch

import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import pickle as pkl


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# images = images.reshape(-1, sequence_length, input_size).to(device)
def flatten(image):
	image = image.reshape(-1, sequence_length, input_size)
	return image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(flatten),
])


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                        shuffle=False)

# print(train_dataset)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        # out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, snip = None, epochs = 20):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

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
        #images = images.reshape(-1, sequence_length, input_size).to(device)
        _, loss = epoch(train_loader, model, criterion,preprocessing=flatten, snip_pruning=snip, optimizer=optimizer)
        # Phase d'evaluation
        with torch.no_grad():
            acc_test, loss_test = epoch(test_loader, model, criterion,preprocessing=flatten)

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
    plt.title("L'erreur moyenne d'un patch dans generation des noms")
    plt.plot(np.arange(len(training_losses_)), training_losses_, label="Train")
    plt.plot(np.arange(len(test_losses_)), test_losses_, label="Test")
    plt.ylabel("L'erreur")
    plt.xlabel('Epochs')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(len(acc_)), acc_)
    plt.show()

# original_model = RNN(input_size, hidden_size, num_layers, num_classes)
# train_losses, test_losses, accuracys  = train_model(original_model)

# import time
# for i in range(3):
# 	original_model = RNN(input_size, hidden_size, num_layers, num_classes)
# 	train_losses, test_losses, accuracys  = train_model(original_model)
# 	pkl.dump([train_losses, test_losses, accuracys], open('org-GRU-MNIST-'+str(i)+'.pkl', 'wb'))
# 	time.sleep(30)

for ratio in [1,5,10,20]:
    prune_model = RNN(input_size, hidden_size, num_layers, num_classes)
    snip = SNIP(prune_model)
    total_param_number = snip.get_total_param_number()
    # print("Original number of params : {}".format(total_param_number))
    K = total_param_number // ratio
    print("nb of params : {}".format(K))

# import time
# for ratio in [10,20,40]:
# 	for i in range(3):
# 		prune_model = RNN(input_size, hidden_size, num_layers, num_classes)
# 		snip = SNIP(prune_model)
# 		total_param_number = snip.get_total_param_number()
# 		print("Original number of params : {}".format(total_param_number))
# 		K = total_param_number // ratio
# 		print("nb of params : {}".format(K))
# 		C_masks = snip.compute_mask(train_loader, K=K)

# 		train_losses, test_losses, accuracys  = train_model(prune_model, snip)
# 		pkl.dump([train_losses, test_losses, accuracys], open(str(ratio)+'-GRU-MNIST-'+str(i)+'.pkl', 'wb'))
# 		time.sleep(30)