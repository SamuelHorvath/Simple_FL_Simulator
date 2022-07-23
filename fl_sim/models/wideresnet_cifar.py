import torch
import torch.nn as nn
import torch.nn.functional as F

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


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0,
                 batch_norm=True):
        super(BasicBlock, self).__init__()
        self.bn1 = create_norm_layer(in_planes, batch_norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = create_norm_layer(out_planes, batch_norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride,
                 dropRate=0.0, batch_norm=True):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers,
            stride, dropRate, batch_norm)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    dropRate, batch_norm):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                          i == 0 and stride or 1, dropRate, batch_norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, dropRate=0.0,
                 batch_norm=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate, batch_norm)
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate, batch_norm)
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate, batch_norm)
        # global average pooling and classifier
        self.bn1 = create_norm_layer(nChannels[3], batch_norm)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(batch_size, -1)
        return self.fc(out)


def WideResNet_28_2(pretrained=False, num_classes=10):
    return WideResNet(28, 2, num_classes=num_classes)


def WideResNet_28_4(pretrained=False, num_classes=10):
    return WideResNet(28, 4, num_classes=num_classes)


def WideResNet_28_8(pretrained=False, num_classes=10):
    return WideResNet(28, 8, num_classes=num_classes)


def WideResNet_28_2_gn(pretrained=False, num_classes=10):
    return WideResNet(28, 2, num_classes=num_classes, batch_norm=False)


def WideResNet_28_4_gn(pretrained=False, num_classes=10):
    return WideResNet(28, 4, num_classes=num_classes, batch_norm=False)


def WideResNet_28_8_gn(pretrained=False, num_classes=10):
    return WideResNet(28, 8, num_classes=num_classes, batch_norm=False)
