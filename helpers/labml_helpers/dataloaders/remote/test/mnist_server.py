import matplotlib.pyplot as plt
from labml import lab
from torchvision import datasets, transforms


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(str(lab.get_data_path()),
                             train=True,
                             download=True,
                             transform=transform)

    RemoteServer

    img = dataset[0]

    plt.imshow(img[0][0], cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
