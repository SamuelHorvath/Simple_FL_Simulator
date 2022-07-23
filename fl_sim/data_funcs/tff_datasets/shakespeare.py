import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .h5_tff_dataset import H5TFFFLDataset, H5TFFDataset

SHAKESPEARE_VOCAB = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?'
                         'bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
SHAKESPEARE_EVAL_BATCH_SIZE = 4


class ShakespeareClient(Dataset):
    def __init__(self, fl_dataset, client_id=None):
        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index=None):
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            if fl.train and len(fl.available_clients) < fl.num_clients:
                fl._add_client_train(index)
            self.length = len(fl.client_and_indices)
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            if fl.train:
                if index not in fl.available_clients:
                    fl._add_client_train(index)
            else:
                raise ValueError(
                    'Individual clients are not supported for test set.')
            self.length = fl.clients_num_data[index]

    def __getitem__(self, index):
        fl = self.fl_dataset
        client, i = fl._get_item_preprocess(self.client_id, index)
        return tuple(tensor[i] for tensor in fl.data[client])

    def __len__(self):
        return self.length


class ShakespeareFL(Dataset):
    """
    Shakespeare Dataset containing dialogs from his books.
    Clients correspond to different characters.
    """
    def __init__(self, data_path, train=True,
                 batch_size=SHAKESPEARE_EVAL_BATCH_SIZE):
        self.train = train
        if train:
            data_path = os.path.join(
                data_path, 'shakespeare/shakespeare_train.h5')
        else:
            data_path = os.path.join(
                data_path, 'shakespeare/shakespeare_test.h5')
        self.batch_size = batch_size
        self.fl_dataset = ShakespeareH5(data_path)
        self.num_clients = self.fl_dataset.num_clients
        self.train = train

        self.available_clients = list()
        self.data = dict()
        self.clients_num_data = dict()
        self.client_and_indices = list()

        if train:
            dataset_s = [ShakespeareH5Client(self.fl_dataset, client_id)
                         for client_id in range(self.num_clients)]
            self.dummy_loader_s = [DataLoader(
                dataset, batch_size=1, shuffle=False)
                for dataset in dataset_s]
            self._add_client_train(None)
        else:
            dataset = ShakespeareH5Client(self.fl_dataset, None)
            self.dummy_loader_s = DataLoader(
                dataset, batch_size=1, shuffle=False)
            self._add_test()

        self.length = len(self.client_and_indices)

    def _add_client_train(self, client_id):
        client_ids = range(self.num_clients) if client_id is None \
             else [client_id]
        for cid in client_ids:
            if cid in self.available_clients:
                continue
            x_data = torch.cat(
                [x[0] for x, y in self.dummy_loader_s[cid]], dim=0)
            y_data = torch.cat(
                [y[0] for x, y in self.dummy_loader_s[cid]], dim=0)
            self._update_data(cid, x_data, y_data)

    def _add_test(self):
        """
        Add test data_preprocess and reshape in a such way
        that subsequent batches correspond
        to the same data_preprocess because of the hidden state.
        :return:
        """
        x_data = torch.cat([x[0] for x, y in self.dummy_loader_s], dim=0)
        y_data = torch.cat([y[0] for x, y in self.dummy_loader_s], dim=0)
        # reorder data_preprocess
        # such that consequent batches follow speech order
        n_zeros = int(np.ceil(len(x_data) / self.batch_size)
                      * self.batch_size) - len(x_data)
        # append zeros if necessary
        x_data = torch.cat([x_data, torch.zeros(
            n_zeros, self.fl_dataset.seq_len).long()], dim=0)
        y_data = torch.cat([y_data, torch.zeros(
            n_zeros, self.fl_dataset.seq_len).long()], dim=0)

        order = np.arange(len(x_data))
        order = order.reshape(self.batch_size, -1).T.reshape(-1)
        x_data, y_data = x_data[order], y_data[order]
        self._update_data(None, x_data, y_data)

    def _update_data(self, cid, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        if self.train:
            self.available_clients.append(cid)
            self.clients_num_data[cid] = x_data.shape[0]
        self.data[cid] = (x_data, y_data)
        self.client_and_indices.extend(
            [(cid, i)for i in range(x_data.shape[0])])

    def _get_item_preprocess(self, client_id, index):
        if client_id is None:
            client, i = self.client_and_indices[index]
        else:
            client, i = client_id, index
        return client, i

    def __len__(self):
        return self.length


class ShakespeareH5(H5TFFFLDataset):
    """
    Preprocessing for Shakespeare h5 Dataset.
    Text to Integer encoding.
    """
    def __init__(self, h5_path, seq_len=80):
        super(ShakespeareH5, self).__init__(
            h5_path, 'shakespeare', 'snippets')
        self.seq_len = seq_len
        # vocabulary
        self.vocab = SHAKESPEARE_VOCAB
        self.char2idx = {u: i for i, u in enumerate(self.vocab, 1)}
        self.idx2char = {i: u for i, u in enumerate(self.vocab, 1)}
        # out of vocabulary, beginning and end of speech
        self.oov = len(self.vocab) + 1
        self.bos = len(self.vocab) + 2
        self.eos = len(self.vocab) + 3


class ShakespeareH5Client(H5TFFDataset):
    def __init__(self, fl_dataset, client_id=None):
        super(ShakespeareH5Client, self).__init__(fl_dataset, client_id)

    def __getitem__(self, index):
        fl = self.fl_dataset
        client, i = fl._get_item_preprocess(self.client_id, index)
        record = fl.dataset[client]['snippets'][i].decode()

        indices = np.array([fl.char2idx[char] if char in fl.char2idx else
                           fl.oov for char in record])
        len_chars = 1 + len(indices)  # beginning of speech
        pad_size = int(np.ceil(
            len_chars/fl.seq_len) * fl.seq_len - len_chars)
        indices = np.concatenate(
            ([fl.bos], indices, [fl.eos], torch.zeros(pad_size)), axis=0)
        x = torch.from_numpy(indices[:-1]).reshape(-1, fl.seq_len)
        y = torch.from_numpy(indices[1:]).reshape(-1, fl.seq_len)
        return x.long(), y.long()
