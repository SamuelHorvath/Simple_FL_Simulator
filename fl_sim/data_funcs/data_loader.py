import torchvision
from torchvision import transforms

from .tff_datasets import FEMNIST, FEMNISTClient, \
     FLCifar100, FLCifar100Client, \
     ShakespeareFL, ShakespeareClient, SHAKESPEARE_EVAL_BATCH_SIZE
from .torch_datasets import FLCifar10, FLCifar10Client


CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def load_data(path, dataset, load_trainset=True, download=True, full_dataset=False):
    dataset = dataset.lower()
    trainsets = None
    fl_dataset = None

    if dataset.startswith("cifar"):  # CIFAR-10/100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ])

        if dataset == "cifar10_fl":
            if load_trainset:
                fl_dataset = FLCifar10(
                    root=path, train=True, download=download,
                    transform=transform_train)
                trainsets = [FLCifar10Client(
                    fl_dataset, client_id=client_id)
                    for client_id in range(fl_dataset.num_clients)]
            testset = torchvision.datasets.CIFAR10(
                root=path, train=False, download=download,
                transform=transform_test)
        elif dataset == "cifar100":
            if load_trainset:
                fl_dataset = FLCifar100(
                    path, train=True, transform=transform_train)
                trainsets = [FLCifar100Client(
                    fl_dataset, client_id=client_id)
                    for client_id in range(fl_dataset.num_clients)]
            fl_dataset = FLCifar100(
                path, train=False, transform=transform_test)
            testset = FLCifar100Client(fl_dataset)
        else:
            raise NotImplementedError(f'{dataset} is not implemented.')
    elif dataset in ["femnist"]:
        if load_trainset:
            fl_dataset = FEMNIST(path, train=True)
            trainsets = [FEMNISTClient(
                fl_dataset, client_id=client_id)
                for client_id in range(fl_dataset.num_clients)]
        fl_dataset = FEMNIST(path, train=False)
        testset = FEMNISTClient(fl_dataset)
    elif dataset in ['shakespeare']:
        if load_trainset:
            fl_dataset = ShakespeareFL(path, train=True)
            trainsets = [ShakespeareClient(fl_dataset, client_id=client_id)
                         for client_id in range(fl_dataset.num_clients)]
        fl_dataset = ShakespeareFL(path, train=False)
        testset = ShakespeareClient(fl_dataset, client_id=None)
    else:
        raise NotImplementedError(f'{dataset} is not implemented.')

    # centralized datasets
    if full_dataset:
        trainsets, testset = [fl_dataset], testset
    return trainsets, testset


def get_test_batch_size(dataset, batch_size):
    if dataset in ['shakespeare']:
        return SHAKESPEARE_EVAL_BATCH_SIZE
    return batch_size


def get_num_classes(dataset):
    dataset = dataset.lower()
    if dataset in ['cifar10']:
        num_classes = 10
    elif dataset in ['cifar100']:
        num_classes = 100
    elif dataset in ['femnist']:
        num_classes = 62
    elif dataset == 'fashion-mnist':
        num_classes = 10
    elif dataset in ['shakespeare']:
        num_classes = 90
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    return num_classes
