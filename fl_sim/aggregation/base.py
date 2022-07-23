"""
Aggregators which takes in weights and gradients.
"""
import torch

from utils.logger import Logger


class _BaseAggregator(object):
    def __init__(self):
        Logger.get().info("Init aggregator: " + self.__str__())
        # log_dict({"Aggregator": self.__str__(), "Type": "Setup"})

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.
        Args:
            inputs (list): A list of tensors to be aggregated.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError
