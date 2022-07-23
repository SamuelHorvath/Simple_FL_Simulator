"""
CNN model for FEMNIST Dataset.
"""
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, channel_1=32, channel_2=64, num_classes=62):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, channel_1, (5, 5))
        self.conv2 = nn.Conv2d(channel_1, channel_2, (5, 5))
        self.fc = nn.Linear(16 * channel_2, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = self.fc(self.flatten(out))
        return out


def simplecnn(pretrained=False, num_classes=62):
    return SimpleCNN(num_classes=num_classes)


def mini_simplecnn(pretrained=False, num_classes=62):
    return SimpleCNN(num_classes=num_classes, channel_1=10, channel_2=20)