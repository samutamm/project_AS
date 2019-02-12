import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LeNet5Caffe(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5Caffe, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 20, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c2', nn.Conv2d(20, 50, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(800, 500)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(500, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

    
class LeNet300100(nn.Module):
    def __init__(self):
        super(LeNet300100, self).__init__()
        self.fc1   = nn.Linear(28*28, 300)
        self.fc2   = nn.Linear(300, 100)
        self.fc3   = nn.Linear(100, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
