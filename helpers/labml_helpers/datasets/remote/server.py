import pickle

import uvicorn
from fastapi import FastAPI, Request, Response
from torch.utils.data import Dataset


class _ServerDataset:
    def __init__(self, name: str, dataset: Dataset):
        self.dataset = dataset
        self.name = name

    def len_handler(self, request: Request):
        sample = pickle.dumps(len(self.dataset))
        return Response(sample, media_type='binary/pickle')

    def item_handler(self, request: Request, idx: str):
        sample = self.dataset[int(idx)]

        sample = pickle.dumps(sample)
        return Response(sample, media_type='binary/pickle')


class DatasetServer:
    r"""
    Remote dataset server

    `Here's a sample usage of the server <https://github.com/labmlai/labml/blob/master/helpers/labml_helpers/datasets/remote/test/mnist_server.py>`_
    """

    def __init__(self):
        self.app = FastAPI()
        self.datasets = {}

    def add_dataset(self, name: str, dataset: Dataset):
        """
        Add a dataset


        Arguments:
            name (str): name of the data set
            dataset (Dataset): dataset to be served
        """
        assert name not in self.datasets
        sd = _ServerDataset(name, dataset)
        self.datasets[name] = sd
        self.app.add_api_route("/" + name + "/len", sd.len_handler, methods=["GET"])
        self.app.add_api_route("/" + name + "/item/{idx}", sd.item_handler, methods=["GET"])

    def start(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Start the server

        Arguments:
            host (str): hostname of the server
            port (int): server port
        """
        uvicorn.run(self.app, host=host, port=port)


def _test():
    from labml import lab
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(str(lab.get_data_path()),
                             train=True,
                             download=True,
                             transform=transform)
    s = DatasetServer()

    s.add_dataset('mnist_train', dataset)

    s.start()


if __name__ == '__main__':
    _test()
