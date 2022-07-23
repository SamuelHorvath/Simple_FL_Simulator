import os
import numpy as np
import h5py

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

TFF_LINK = 'https://storage.googleapis.com/tff-datasets-public/'

TFF_DATASETS = {
    'cifar100_fl':
        TFF_LINK + 'fed_cifar100.tar.bz2',
    'femnist':
        TFF_LINK + 'fed_emnist.tar.bz2',
    'shakespeare':
        TFF_LINK + 'shakespeare.tar.bz2',
}


class H5TFFDataset(Dataset):
    """
    Based FL class that loads H5 type data_preprocess.
    """
    def __init__(self, fl_dataset, client_id):
        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index=None):
        """
        Set pointer to client's data_preprocess corresponding to index
        :param index: index of client
        :return:
        """
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            self.length = len(fl.client_s)
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = fl.clients_num_data[index]

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.length


class H5TFFFLDataset:
    """
    Based FL class that loads H5 type data_preprocess.
    """
    def __init__(self, h5_path, dataset_name, data_key, download=True):
        self.h5_path = h5_path
        if not os.path.isfile(h5_path):
            one_up = os.path.dirname(h5_path)
            target = os.path.basename(TFF_DATASETS[dataset_name])
            if download:
                download_url(TFF_DATASETS[dataset_name], one_up)

            def extract_bz2(filename, path="."):
                import tarfile
                with tarfile.open(filename, "r:bz2") as tar:
                    tar.extractall(path)
            taret_file = os.path.join(one_up, target)
            if os.path.isfile(taret_file):
                extract_bz2(os.path.join(one_up, target), one_up)
            else:
                raise ValueError(f"{taret_file}: does not exists,"
                                 f" set `download=True`.")

        self.dataset = None
        self.clients = list()
        self.clients_num_data = list()
        self.client_s = list()
        self.index_s = list()
        with h5py.File(self.h5_path, 'r') as file:
            data = file['examples']
            for client in list(data.keys()):
                self.clients.append(client)
                n_data = len(data[client][data_key])
                for i in range(n_data):
                    self.index_s.append(i)
                    self.client_s.append(client)
                self.clients_num_data.append(n_data)
        self.clients = np.array(self.clients).astype(np.string_)
        self.clients_num_data = np.array(self.clients_num_data)
        self.client_s = np.array(self.client_s).astype(np.string_)
        self.index_s = np.array(self.index_s)
        self.num_clients = len(self.clients)
        self.length = len(self.client_s)

    def _get_item_preprocess(self, client_id, index):
        # loading in getitem allows us to
        # use multiple processes for data_preprocess loading
        # because hdf5 files aren't pickelable
        #  so can't transfer them across processes
        # https://discuss.pytorch.org/t/hdf5-a-data-format-for-pytorch/40379
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')["examples"]
        if client_id is None:
            client, i = self.client_s[index], self.index_s[index]
        else:
            client, i = self.clients[client_id], index
        return client, i

    def __len__(self):
        return self.length
