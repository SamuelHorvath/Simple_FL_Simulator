from numpy.random import default_rng


from .base import _ClientSampler


class UniformSampler(_ClientSampler):
    def __init__(self, comm_rounds, num_clients_per_round, num_clients, seed):
        super(UniformSampler, self).__init__()
        rng = default_rng(seed)
        self.sampled_clients = [rng.choice(
            num_clients, num_clients_per_round, replace=False)
            for _ in range(comm_rounds)]

    def __str__(self):
        return "Uniform Sampler"

    def get_sampled_clients(self, comm_round):
        return self.sampled_clients[comm_round - 1]
