# From
# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = [
    'ResNet', 'resnet20', 'resnet32', 'resnet44',
    'resnet56', 'resnet110', 'resnet1202']


GROUP_NORM_LOOKUP = {
    16: 2,  # -> channels per group: 8
    32: 4,  # -> channels per group: 8
    64: 8,  # -> channels per group: 8
    128: 8,  # -> channels per group: 16
    256: 16,  # -> channels per group: 16
    512: 32,  # -> channels per group: 16
    1024: 32,  # -> channels per group: 32
    2048: 32,  # -> channels per group: 64
}


def create_norm_layer(num_channels, batch_norm=True):
    if batch_norm:
        return nn.BatchNorm2d(num_channels)
    return nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 option='A', batch_norm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = create_norm_layer(planes, batch_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = create_norm_layer(planes, batch_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (
                        0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, self.expansion * planes,
                        kernel_size=1, stride=stride, bias=False),
                    create_norm_layer(self.expansion * planes, batch_norm)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, batch_norm=True):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = create_norm_layer(16, batch_norm)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1, batch_norm=batch_norm)
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2, batch_norm=batch_norm)
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2, batch_norm=batch_norm)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, batch_norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride, batch_norm=batch_norm))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes)


def resnet20gn(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3],
                  num_classes=num_classes, batch_norm=False)


def resnet32gn(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5],
                  num_classes=num_classes, batch_norm=False)


def resnet44gn(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7],
                  num_classes=num_classes, batch_norm=False)


def resnet56gn(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9],
                  num_classes=num_classes, batch_norm=False)


def resnet110gn(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18],
                  num_classes=num_classes, batch_norm=False)


def resnet1202gn(pretrained=False, num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200],
                  num_classes=num_classes, batch_norm=False)
