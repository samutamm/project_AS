
from time import sleep
import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import EMNIST
import torchvision.transforms as transforms

from models.LeNet import LeNet300100
from Snip_copy import SNIP
from optimization import MeanEvaluator

class SubSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, label):
        self.indices = np.where(dataset.train_labels.numpy() == label)[0]

    def __len__ (self):
        return len(self.indices)

    def __iter__(self):
        np.random.shuffle(self.indices)
        return iter(torch.from_numpy(self.indices))

class NLayerParameterVisualizator:

    def __init__(self, dataset, labels):
        dataloaders = []
        for label in labels:
            sampler = SubSampler(dataset, label)
            dataloaders.append(torch.utils.data.DataLoader(dataset,
                                                              batch_size=100,
                                                              num_workers=2,
                                                              sampler=sampler))
        self.dataloaders = dataloaders

    def train(self):
        for i, dataloader in enumerate(self.dataloaders):
            for j, pr in enumerate(np.linspace(0.1, 0.9, 9)):
                evaluator = MeanEvaluator(LeNet300100,
                                          dataset = None,
                                          eval_n = 0,
                                          epochs=15,
                                          pruning_ratio=pr)

                evaluator.pruning_data_loader = dataloader
                evaluator.train_data_loader = dataloader
                evaluator.test_data_loader = dataloader

                prune_model, snip = evaluator.create_pruning_model()
                _, test_losses, accuracys = evaluator.train_model(prune_model, snip)
                print(np.max(accuracys))

                with torch.no_grad():
                    weights = prune_model.fc1.weight.detach().cpu().numpy()
                    weights = weights.mean(axis=0).reshape(28,28)
                    # stock locally and retrieve later
                    filename = "./src/images/" + str(i) + "_" + str(j + 1) + ".npy"
                    with open(filename, 'wb') as f:
                        np.save(f, weights)

                sleep(5)

    def combine_images(self):
        fig = plt.figure(figsize=(8, 8))
        columns = len(self.dataloaders)
        rows = 9

        for i in range(1,columns+1):
            for j in range(1,rows+1):
                filename = "./src/images/" + str(i-1) + "_" + str(j) + ".npy"
                img = np.load(filename)
                image_i = i + columns * (j-1)
                fig.add_subplot(rows, columns, image_i)
                plt.imshow(img)
        plt.show()

    def sample_images(self):
        fig = plt.figure(figsize=(8, 8))
        columns = len(self.dataloaders)
        rows = 9

        examples = 2

        for i, dataloader in enumerate(self.dataloaders):
            X, y = next(iter(dataloader))
            indexes = np.random.choice(X.shape[0], size=rows, replace=False)

            for j in range(rows + examples):
                if j < examples:
                    img = X[indexes[j], 0].reshape(28,28).numpy()
                else:
                    filename = "./src/images/" + str(i) + "_" + str(j-examples +1 ) + ".npy"
                    img = np.load(filename)

                image_i = 1 + i + columns * j
                ax = fig.add_subplot(rows + examples, columns, image_i)
                ax.axis("off")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(img)
        plt.subplots_adjust(wspace=0, hspace=0.1)
        plt.show()


if __name__ == '__main__':

    dataset = EMNIST('./data', train=True, download=True, split="byclass",
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    #viz = NLayerParameterVisualizator(dataset, list(range(2,10)))
    viz = NLayerParameterVisualizator(dataset, list(range(2, 8)))

    viz.sample_images()
    #viz.train()
    #viz.combine_images()


