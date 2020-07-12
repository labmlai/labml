import torch

from labml.logger import inspect


def test():
    inspect("test")
    inspect([1, 2])
    inspect({'x': 2, 4: '5'})
    small_torch = torch.arange(0, 10)
    large_torch = torch.arange(0, 1000).view(10, 100)
    small_np = small_torch.numpy()
    large_np = large_torch.numpy()
    inspect(small_np)
    inspect(large_np)
    inspect({'small': small_np, 'large': large_np})
    inspect({'small': small_torch, 'large': large_torch})
    inspect(small_torch)

    inspect(torch.tensor(small_np))


if __name__ == '__main__':
    test()
