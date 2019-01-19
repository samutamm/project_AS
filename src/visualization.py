
import torch
from torch import nn

import numpy as np

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
        pass
