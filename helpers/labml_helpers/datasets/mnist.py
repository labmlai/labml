from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from labml import lab
from labml.configs import BaseConfigs, aggregate, option


def _dataset(is_train, transform):
    return datasets.MNIST(str(lab.get_data_path()),
                          train=is_train,
                          download=True,
                          transform=transform)


class MNISTConfigs(BaseConfigs):
    """
    Configurable MNIST data set.

    Arguments:
        dataset_name (str): name of the data set, ``MNIST``
        dataset_transforms (torchvision.transforms.Compose): image transformations
        train_dataset (torchvision.datasets.MNIST): training dataset
        valid_dataset (torchvision.datasets.MNIST): validation dataset

        train_loader (torch.utils.data.DataLoader): training data loader
        valid_loader (torch.utils.data.DataLoader): validation data loader

        train_batch_size (int): training batch size
        valid_batch_size (int): validation batch size

        train_loader_shuffle (bool): whether to shuffle training data
        valid_loader_shuffle (bool): whether to shuffle validation data
    """

    dataset_name: str = 'MNIST'
    dataset_transforms: transforms.Compose
    train_dataset: datasets.MNIST
    valid_dataset: datasets.MNIST

    train_loader: DataLoader
    valid_loader: DataLoader

    train_batch_size: int = 64
    valid_batch_size: int = 1024

    train_loader_shuffle: bool = True
    valid_loader_shuffle: bool = False


@option(MNISTConfigs.dataset_transforms)
def mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


@option(MNISTConfigs.train_dataset)
def mnist_train_dataset(c: MNISTConfigs):
    return _dataset(True, c.dataset_transforms)


@option(MNISTConfigs.valid_dataset)
def mnist_valid_dataset(c: MNISTConfigs):
    return _dataset(False, c.dataset_transforms)


@option(MNISTConfigs.train_loader)
def mnist_train_loader(c: MNISTConfigs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=c.train_loader_shuffle)


@option(MNISTConfigs.valid_loader)
def mnist_valid_loader(c: MNISTConfigs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=c.valid_loader_shuffle)


aggregate(MNISTConfigs.dataset_name, 'MNIST',
          (MNISTConfigs.dataset_transforms, 'mnist_transforms'),
          (MNISTConfigs.train_dataset, 'mnist_train_dataset'),
          (MNISTConfigs.valid_dataset, 'mnist_valid_dataset'),
          (MNISTConfigs.train_loader, 'mnist_train_loader'),
          (MNISTConfigs.valid_loader, 'mnist_valid_loader'))
