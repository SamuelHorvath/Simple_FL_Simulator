"""
Sampler to control client distribution.
"""
from utils.logger import Logger


class _ClientSampler(object):
    def __init__(self):
        Logger.get().info("Init sampler: " + self.__str__())

    def get_sampled_clients(self, comm_round):
        """Aggregate the inputs and update in-place.
        Args:
            comm_round (int): communication round for which to get sampled clients.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError
