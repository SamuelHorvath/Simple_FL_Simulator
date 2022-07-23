import os
import numpy as np
import torchvision

from .h5_tff_dataset import H5TFFDataset, H5TFFFLDataset


class FEMNISTClient(H5TFFDataset):
    def __init__(self, fl_dataset, client_id=None):
        super(FEMNISTClient, self).__init__(fl_dataset, client_id)

    def __getitem__(self, index):
        fl = self.fl_dataset
        client, i = fl._get_item_preprocess(self.client_id, index)
        x = 1 - fl.transform(fl.dataset[client]['pixels'][i])
        y = np.int64(fl.dataset[client]['label'][i])
        return x, y


class FEMNIST(H5TFFFLDataset):
    """
    Federated Extended MNIST Dataset.
    Clients corresponds to different person handwriting.
    """
    def __init__(self, h5_path, train=True):
        if train:
            h5_path = os.path.join(h5_path, 'femnist/fed_emnist_train.h5')
        else:
            h5_path = os.path.join(h5_path, 'femnist/fed_emnist_test.h5')
        super(FEMNIST, self).__init__(h5_path, 'femnist', 'pixels')
        self.transform = torchvision.transforms.ToTensor()
