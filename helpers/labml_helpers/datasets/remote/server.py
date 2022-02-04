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
    def __init__(self):
        self.app = FastAPI()
        self.datasets = {}

    def add_dataset(self, name: str, dataset: Dataset):
        assert name not in self.datasets
        sd = _ServerDataset(name, dataset)
        self.datasets[name] = sd
        self.app.add_api_route("/" + name + "/len", sd.len_handler, methods=["GET"])
        self.app.add_api_route("/" + name + "/item/{idx}", sd.item_handler, methods=["GET"])

    def start(self, host="0.0.0.0", port=8000):
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
