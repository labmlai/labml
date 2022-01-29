import pickle

import uvicorn
from fastapi import FastAPI, Request, Response
from torch.utils.data import Dataset


class RemoteServer:
    def __init__(self, dataset: Dataset, host="0.0.0.0", port=8000):
        self.dataset = dataset
        self.app = FastAPI()
        self.app.add_api_route("/len", self.len_handler, methods=["GET"])
        self.app.add_api_route("/item/{idx}", self.item_handler, methods=["GET"])
        uvicorn.run(self.app, host=host, port=port)

    def len_handler(self, request: Request):
        sample = pickle.dumps(len(self.dataset))
        return Response(sample, media_type='binary/pickle')

    def item_handler(self, request: Request, idx: str):
        sample = self.dataset[int(idx)]
        print(idx)

        sample = pickle.dumps(sample)
        return Response(sample, media_type='binary/pickle')


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
    s = RemoteServer(dataset)


if __name__ == '__main__':
    _test()
