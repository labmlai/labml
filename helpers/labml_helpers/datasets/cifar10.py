from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from labml import lab
from labml.configs import BaseConfigs


def _dataset(is_train, transform):
    return datasets.CIFAR10(str(lab.get_data_path()),
                            train=is_train,
                            download=True,
                            transform=transform)


class CIFAR10Configs(BaseConfigs):
    dataset_name: str = 'CIFAR10'
    dataset_transforms: transforms.Compose
    train_dataset: datasets.CIFAR10
    valid_dataset: datasets.CIFAR10

    train_loader: DataLoader
    valid_loader: DataLoader

    train_batch_size: int = 64
    valid_batch_size: int = 1024

    train_loader_shuffle: bool = True
    valid_loader_shuffle: bool = False


@CIFAR10Configs.calc(CIFAR10Configs.dataset_transforms)
def cifar10_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


@CIFAR10Configs.calc(CIFAR10Configs.train_dataset)
def cifar10_train_dataset(c: CIFAR10Configs):
    return _dataset(True, c.dataset_transforms)


@CIFAR10Configs.calc(CIFAR10Configs.valid_dataset)
def cifar10_valid_dataset(c: CIFAR10Configs):
    return _dataset(False, c.dataset_transforms)


@CIFAR10Configs.calc(CIFAR10Configs.train_loader)
def cifar10_train_loader(c: CIFAR10Configs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=c.train_loader_shuffle)


@CIFAR10Configs.calc(CIFAR10Configs.valid_loader)
def cifar10_valid_loader(c: CIFAR10Configs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=c.valid_loader_shuffle)


CIFAR10Configs.aggregate(CIFAR10Configs.dataset_name, 'CIFAR10',
                       (CIFAR10Configs.dataset_transforms, 'cifar10_transforms'),
                       (CIFAR10Configs.train_dataset, 'cifar10_train_dataset'),
                       (CIFAR10Configs.valid_dataset, 'cifar10_valid_dataset'),
                       (CIFAR10Configs.train_loader, 'cifar10_train_loader'),
                       (CIFAR10Configs.valid_loader, 'cifar10_valid_loader'))
