import os
from PIL import Image
from .h5_tff_dataset import H5TFFDataset, H5TFFFLDataset


class FLCifar100Client(H5TFFDataset):
    def __init__(self, fl_dataset, client_id=None):
        super(FLCifar100Client, self).__init__(fl_dataset, client_id)

    def __getitem__(self, index):
        fl = self.fl_dataset
        client, i = fl._get_item_preprocess(self.client_id, index)
        img = Image.fromarray(fl.dataset[client]['image'][i])
        x = fl.transform(img)
        y = fl.dataset[client]['label'][i]
        return x, y


class FLCifar100(H5TFFFLDataset):
    """
    CIFAR100 Dataset.
    500 clients that were allocated data_preprocess using LDA.
    """
    def __init__(self, h5_path, transform, train=True):
        if train:
            h5_path = os.path.join(
                h5_path, 'cifar100_fl/fed_cifar100_train.h5')
        else:
            h5_path = os.path.join(
                h5_path, 'cifar100_fl/fed_cifar100_test.h5')

        super(FLCifar100, self).__init__(
            h5_path, 'cifar100_fl', 'image')
        self.transform = transform
