from labml import lab
from labml_helpers.datasets.remote import DatasetServer
from torchvision import datasets, transforms


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(str(lab.get_data_path()),
                                   train=True,
                                   download=True,
                                   transform=transform)

    valid_dataset = datasets.MNIST(str(lab.get_data_path()),
                                   train=False,
                                   download=True,
                                   transform=transform)

    ds = DatasetServer()
    ds.add_dataset('mnist_train', train_dataset)
    ds.add_dataset('mnist_valid', valid_dataset)

    ds.start()


if __name__ == '__main__':
    main()
