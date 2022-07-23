from torch import nn, optim

from models import *
from data_funcs import *
from aggregation import *
from client_sampling import *


def get_task_elements(task: str, test_batch_size, data_path):
    if task == 'femnist':
        train_sets, test_set = load_data(dataset='femnist', path=data_path)
        return simplecnn(), nn.CrossEntropyLoss(), False,\
            get_test_batch_size('femnist', test_batch_size), train_sets, test_set
    if task == 'shakespeare':
        train_sets, test_set = load_data(dataset='shakespeare', path=data_path)
        return simple_rnn(), nn.CrossEntropyLoss(), True,\
            get_test_batch_size('shakespeare', test_batch_size), train_sets, test_set
    # ADD YOUR NEW TASK HERE, IF YOU ADD NEW MODEL OR DATASET,
    #  EDIT MODELS OR DATA FUNCTIONS TO IMPORT NEW FUNCTIONALITY
    raise ValueError(f"Task \"{task}\" does not exists.")


def get_agg(aggregation: str):
    if aggregation == 'mean':
        return Mean()
    # ADD YOUR NEW AGGREGATION HERE, EDIT AGGREGATION MODULE TO IMPORT NEW FUNCTIONALITY
    raise ValueError(f"Aggregation \"{aggregation}\" does not exists.")


def get_sampling(sampling: str, comm_rounds: int, num_clients_per_round: int, num_clients: int, seed: int):
    if sampling == 'uniform':
        return UniformSampler(comm_rounds, num_clients_per_round, num_clients, seed)
    # ADD YOUR NEW SAMPLING HERE, EDIT AGGREGATION MODULE TO IMPORT NEW FUNCTIONALITY
    raise ValueError(f"Sampling \"{sampling}\" is not defined.")


def get_optimizer_init(optimizer, lr):
    # ADD YOUR NEW OPTIMIZERS HERE
    if optimizer == 'sgd':
        def sgd_init(params):
            return optim.SGD(params, lr=lr)
        return sgd_init
    else:
        raise ValueError(f"Optimizer \"{optim}\" is not defined.")
