import torch

from .base import _BaseAggregator


class Mean(_BaseAggregator):
    def __call__(self, inputs, *args, **kwargs):
        values = torch.stack(inputs, dim=0).mean(dim=0)
        return values

    def __str__(self):
        return "Mean"
