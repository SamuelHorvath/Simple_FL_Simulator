from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from ..utils import partition_data_based_on_labels


class FLCifar10Client(Dataset):
    def __init__(self, fl_dataset, client_id=None):

        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index=None):
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            self.length = len(fl.data)
            self.data = fl.data
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            indices = fl.partition[self.client_id]
            self.length = len(indices)
            self.data = fl.data[indices]
            self.targets = [fl.targets[i] for i in indices]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        fl = self.fl_dataset
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other fl_datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if fl.transform is not None:
            img = fl.transform(img)

        if fl.target_transform is not None:
            target = fl.target_transform(target)

        return img, target

    def __len__(self):
        return self.length


class FLCifar10(CIFAR10):
    """
    CIFAR10 Dataset.
    100 clients that were allocated data_preprocess uniformly at random.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(FLCifar10, self).__init__(root, train=train, transform=transform,
                                        target_transform=target_transform,
                                        download=download)

        # self.partition = partition_data_based_on_labels(dataset=self.data, n_clients=100)
        # Uniform shuffle
        shuffle = np.arange(len(self.data))
        rng = np.random.default_rng(12345)
        rng.shuffle(shuffle)
        self.partition = shuffle.reshape([100, -1])
        self.num_clients = len(self.partition)
