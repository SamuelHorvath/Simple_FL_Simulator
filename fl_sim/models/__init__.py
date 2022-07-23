# ResNets for CIFAR Datasets
from .resnet_cifar import resnet20
from .resnet_cifar import resnet32
from .resnet_cifar import resnet44
from .resnet_cifar import resnet56
from .resnet_cifar import resnet110
from .resnet_cifar import resnet1202

from .resnet_cifar import resnet20gn
from .resnet_cifar import resnet32gn
from .resnet_cifar import resnet44gn
from .resnet_cifar import resnet56gn
from .resnet_cifar import resnet110gn
from .resnet_cifar import resnet1202gn

from .resnet_cifarlike import ResNet18 as resnet18_cifar
from .resnet_cifarlike import ResNet34 as resnet34_cifar
from .resnet_cifarlike import ResNet50 as resnet50_cifar
from .resnet_cifarlike import ResNet101 as resnet101_cifar
from .resnet_cifarlike import ResNet152 as resnet152_cifar

from .resnet_cifarlike import ResNet18gn as resnet18gn_cifar
from .resnet_cifarlike import ResNet34gn as resnet34gn_cifar
from .resnet_cifarlike import ResNet50gn as resnet50gn_cifar
from .resnet_cifarlike import ResNet101gn as resnet101gn_cifar
from .resnet_cifarlike import ResNet152gn as resnet152gn_cifar

# VGGs for CIFAR Datasets
from .vgg_cifar import vgg11 as vgg11_cifar
from .vgg_cifar import vgg13 as vgg13_cifar
from .vgg_cifar import vgg16 as vgg16_cifar
from .vgg_cifar import vgg19 as vgg19_cifar

from .vgg_cifar import vgg11gn as vgg11gn_cifar
from .vgg_cifar import vgg13gn as vgg13gn_cifar
from .vgg_cifar import vgg16gn as vgg16gn_cifar
from .vgg_cifar import vgg19gn as vgg19gn_cifar

# WideResNets for CIFAR Datasets
from .wideresnet_cifar import WideResNet_28_2 as wideresnet282_cifar
from .wideresnet_cifar import WideResNet_28_4 as wideresnet284_cifar
from .wideresnet_cifar import WideResNet_28_8 as wideresnet288_cifar

from .wideresnet_cifar import WideResNet_28_2_gn as wideresnet282gn_cifar
from .wideresnet_cifar import WideResNet_28_4_gn as wideresnet284gn_cifar
from .wideresnet_cifar import WideResNet_28_8_gn as wideresnet288gn_cifar

# models for Shakespeare
from .simple_rnn import simple_rnn, mini_simple_rnn

# models for FEMNIST
from .simple_cnn import simplecnn, mini_simplecnn

# model for synthetic experiments
from .log_reg import logreg

__all__ = [
    'resnet20', 'resnet32', 'resnet44',
    'resnet56', 'resnet110', 'resnet1202',
    'resnet20gn', 'resnet32gn', 'resnet44gn',
    'resnet56gn', 'resnet110gn', 'resnet1202gn',
    'resnet18_cifar', 'resnet34_cifar', 'resnet50_cifar',
    'resnet101_cifar', 'resnet152_cifar',
    'resnet18gn_cifar', 'resnet34gn_cifar', 'resnet50gn_cifar',
    'resnet101gn_cifar', 'resnet152gn_cifar',
    'vgg11_cifar', 'vgg13_cifar',
    'vgg16_cifar', 'vgg19_cifar',
    'vgg11gn_cifar', 'vgg13gn_cifar',
    'vgg16gn_cifar', 'vgg19gn_cifar',
    'wideresnet282_cifar', 'wideresnet284_cifar',
    'wideresnet288_cifar',
    'wideresnet282gn_cifar', 'wideresnet284gn_cifar',
    'wideresnet288gn_cifar',
    'simplecnn', 'mini_simplecnn',
    'simple_rnn', 'mini_simple_rnn',
    'log_reg',
    ]
