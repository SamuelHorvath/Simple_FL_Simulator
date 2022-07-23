# import numpy as np
# from numpy.random import default_rng
# import torch
# from copy import deepcopy
# from .logger import Logger


# def get_sampled_clients(comm_rounds, num_clients_per_round, num_clients, seed):
#     rng = default_rng(seed)
#     sampled_clients = [rng.choice(
#         num_clients, num_clients_per_round, replace=False)
#         for _ in range(comm_rounds)]
#     return sampled_clients

# def get_sampled_clients(client_distribution, train_sets, epochs, seed):
#     num_clients_per_round = int(client_distribution.split('_')[1])
#     sample_size = len(train_sets)
#     client_distribution_name = client_distribution.split('_')[0]
#     assert num_clients_per_round <= sample_size, \
#         "Number of clients per round is more that all clients."
#     random.seed(seed)
#     # clients are pre-sampled for deterministic participation among runs
#     if client_distribution_name == 'uniform':
#         prob_clients = np.ones(
#             sample_size) * (num_clients_per_round / sample_size)
#         sampled_clients = [random.choice(
#             sample_size, num_clients_per_round, replace=False)
#             for _ in range(epochs)]
#     elif client_distribution_name == 'data':
#         num_data = np.array([len(dataset) for dataset in train_sets])
#         prob_clients = get_prob_data(num_data, num_clients_per_round)
#         sampled_clients = [np.where(
#             random.uniform(size=sample_size) < prob_clients)[0]
#                            for _ in range(epochs)]
#     else:
#         raise ValueError(
#             f"Client dist. {client_distribution_name} was not recognised.")
#     return sampled_clients, prob_clients
#
#
# def get_sampled_local_epochs(local_sampler, sampled_clients, epochs, seed):
#     np.random.seed(seed)
#     # epochs are pre-sampled for deterministic participation among runs
#     local_sampler_name = local_sampler.split('_')[0]
#     if local_sampler_name == 'static':
#         local_epochs = int(local_sampler.split('_')[1])
#         sampled_epochs = [
#             np.ones(len(sampled_clients[i]), dtype=int) * local_epochs
#             for i in range(epochs)]
#     elif local_sampler_name == 'unif':
#         min_epochs = int(local_sampler.split('_')[1])
#         max_epochs = int(local_sampler.split('_')[2])
#         sampled_epochs = [random.randint(
#             min_epochs, max_epochs + 1, len(sampled_clients[i]))
#                           for i in range(epochs)]
#     else:
#         raise ValueError(
#             f"Local sampler {local_sampler_name} was not recognised.")
#     return sampled_epochs
#
#
# def get_prob_data(num_data, num_clients, iters=20):
#     prob = num_clients * num_data / sum(num_data)
#     all_ones = np.ones(num_data.shape)
#     for _ in range(iters):
#         prob *= num_clients / np.sum(prob)
#         prob = np.min([prob, all_ones], axis=0)
#         if np.sum(prob) == num_clients:
#             break
#     return prob
#
#
# @torch.no_grad()
# def update_train_dicts(state_dicts, weights):
#     # get dictionary structure
#     model_dict = deepcopy(state_dicts[0]['model'])
#     optimiser_dict = deepcopy(state_dicts[0]['optimiser'])
#
#     # model state_dict (structure layer key: value)
#     Logger.get().info('Aggregating model state dict.')
#     for layer in model_dict:
#         layer_vals = torch.stack(
#             [state_dict['model'][layer] for state_dict in state_dicts])
#         model_dict[layer] = weighted_sum(layer_vals, weights)
#
#     # optimiser state dict
#     # (structure: layer key (numeric): buffers for layer: value)
#     # normalize weights for state
#     if 'state' in optimiser_dict:
#         Logger.get().info('Aggregating optimiser state dict.')
#         for l_key in optimiser_dict['state']:
#             layer = optimiser_dict['state'][l_key]
#             for buffer in layer:
#                 if layer[buffer] is not None:
#                     buffer_vals = torch.stack(
#                         [state_dict['optimiser']['state'][l_key][buffer]
#                             for state_dict in state_dicts])
#                     optimiser_dict['state'][l_key][buffer] = weighted_sum(
#                         buffer_vals, weights / sum(weights))
#     return model_dict, optimiser_dict
#
#
# def weighted_sum(tensors, weights):
#     extra_dims = (1,)*(tensors.dim()-1)
#     return torch.sum(weights.view(-1, *extra_dims) * tensors, dim=0)
