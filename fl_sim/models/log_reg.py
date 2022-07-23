"""
Logistic Regression.
"""
from torch import nn


class LogReg(nn.Module):
    def __init__(self, input_dim=60, num_classes=10):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out


def logreg(pretrained=False, num_classes=10):
    return LogReg(num_classes=num_classes)