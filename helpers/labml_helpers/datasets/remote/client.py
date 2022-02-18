import pickle

import matplotlib.pyplot as plt
import urllib3
from torch.utils.data import Dataset


class RemoteDataset(Dataset):
    """
    Remote dataset

    Arguments:
        name (str): name of the data set, as specified in
         :class:`labml_helpers.datasets.remote.DatasetServer`
        host (str): hostname of the server
        post (int): port of the server

    `Here's a sample <https://github.com/labmlai/labml/blob/master/helpers/labml_helpers/datasets/remote/test/mnist_train.py>`_
    """

    def __init__(self, name: str, host: str = "0.0.0.0", port: int = 8000):
        self.name = name
        self.port = port
        self.host = host
        self.http = urllib3.PoolManager()
        self._len = None

    def __getitem__(self, item):
        r = self.http.request('GET', f'http://{self.host}:{self.port}/{self.name}/item/{item}')
        return pickle.loads(r.data)

    def __len__(self):
        if self._len is None:
            r = self.http.request('GET', f'http://{self.host}:{self.port}/{self.name}/len')
            self._len = pickle.loads(r.data)

        return self._len


def _test():
    dataset = RemoteDataset('mnist_train')
    print(len(dataset))
    img = dataset[0]

    plt.imshow(img[0][0], cmap='gray')
    plt.show()


if __name__ == '__main__':
    _test()
