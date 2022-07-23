"""
VGG11/13/16/19 in Pytorch.
"""
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
              512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
              512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
              'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

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


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, batch_norm=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], batch_norm)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           create_norm_layer(x, batch_norm),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(pretrained=False, num_classes=10):
    return VGG('VGG11', num_classes=num_classes)


def vgg13(pretrained=False, num_classes=10):
    return VGG('VGG13', num_classes=num_classes)


def vgg16(pretrained=False, num_classes=10):
    return VGG('VGG16', num_classes=num_classes)


def vgg19(pretrained=False, num_classes=10):
    return VGG('VGG19', num_classes=num_classes)


def vgg11gn(pretrained=False, num_classes=10):
    return VGG('VGG11', num_classes=num_classes, batch_norm=False)


def vgg13gn(pretrained=False, num_classes=10):
    return VGG('VGG13', num_classes=num_classes, batch_norm=False)


def vgg16gn(pretrained=False, num_classes=10):
    return VGG('VGG16', num_classes=num_classes, batch_norm=False)


def vgg19gn(pretrained=False, num_classes=10):
    return VGG('VGG19', num_classes=num_classes, batch_norm=False)
