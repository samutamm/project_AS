
from optimization import MeanEvaluator
from models.LeNet import LeNet5Caffe

import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def fit_random_labels(original_dataset, dataset):
    epochs = 300
    eval_true_labels = MeanEvaluator(LeNet5Caffe,
                          dataset = MNIST,
                          eval_n = 1,
                          epochs=epochs,
                          pruning_ratio=1)

    true_label_resultats = eval_true_labels.baseline_training()

    # CREATE RANDOM EVALUATOR
    eval_random_labels = MeanEvaluator(LeNet5Caffe,
                         dataset=None,
                         eval_n=1,
                         epochs=epochs,
                         pruning_ratio=0.05)
    eval_random_labels.pruning_data_loader = torch.utils.data.DataLoader(original_dataset,batch_size=100,num_workers=2)
    eval_random_labels.train_data_loader = torch.utils.data.DataLoader(dataset,batch_size=100,num_workers=2)
    # No need of test data loader, because it is about fitting the training data
    eval_random_labels.skip_test_evaluation = True

    random_label_noprune = eval_random_labels.baseline_training()
    random_label_prune = eval_random_labels.snip_training()

    plt.plot(np.arange(epochs), true_label_resultats[0], label="true label")
    plt.plot(np.arange(epochs), random_label_noprune[0], label="random label (no prune)")
    plt.plot(np.arange(epochs), random_label_prune[0], label="random label (prune)")
    print(true_label_resultats[0])
    print(random_label_noprune[0])
    print(random_label_prune[0])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    original_dataset = MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    dataset = MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
    random_order = torch.randperm(dataset.train_labels.shape[0])
    dataset.train_labels = dataset.train_labels[random_order]
    fit_random_labels(original_dataset, dataset)

