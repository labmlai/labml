import pickle

import matplotlib.pyplot as plt
import urllib3
from torch.utils.data import Dataset


class RemoteDataset(Dataset):
    def __init__(self, host="0.0.0.0", port=8000):
        self.port = port
        self.host = host
        self.http = urllib3.PoolManager()
        self._len = None

    def __getitem__(self, item):
        r = self.http.request('GET', f'http://{self.host}:{self.port}/item/{item}')
        return pickle.loads(r.data)

    def __len__(self):
        if self._len is None:
            r = self.http.request('GET', f'http://{self.host}:{self.port}/len')
            self._len = pickle.loads(r.data)

        return self._len


def _test():
    dataset = RemoteDataset()
    print(len(dataset))
    img = dataset[0]

    plt.imshow(img[0][0], cmap='gray')
    plt.show()


if __name__ == '__main__':
    _test()
