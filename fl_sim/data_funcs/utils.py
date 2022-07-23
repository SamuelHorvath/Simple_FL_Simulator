import torch
import numpy as np


def partition_data_based_on_labels(dataset, n_clients=3, random_seed=1, alpha=0.1):
    y_s = torch.tensor([dataset.__getitem__(i)[1] for i in range(len(dataset))])

    labels = torch.unique(y_s)
    n_classes = len(labels)

    np.random.seed(random_seed)
    dist = np.random.dirichlet(np.ones(n_clients) * alpha, n_classes)

    labels_ind = {
        j: [] for j in range(n_classes)
    }

    labels_map = {
        label.item(): j for j, label in enumerate(labels)
    }

    for i, y in enumerate(y_s):
        labels_ind[labels_map[y.item()]].append(i)

    labels_len = np.array([len(labels_ind[j]) for j in range(n_classes)])

    dist = np.cumsum(dist, axis=1) * labels_len.reshape(n_classes, 1)
    dist = dist.astype(int)
    dist = np.concatenate([np.zeros((n_classes, 1), int), dist], axis=1)

    clients_ind = {
        i: [] for i in range(n_clients)
    }

    for i in range(n_clients):
        for j in range(n_classes):
            clients_ind[i].extend(labels_ind[j][dist[j, i]:dist[j, i + 1]])
    return clients_ind
