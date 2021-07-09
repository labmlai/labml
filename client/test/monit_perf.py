import time

import torch

from labml import monit, logger
from labml.logger import Text

N = 10_000


def no_section():
    arr = torch.zeros((1000, 1000))

    for i in range(N):
        for t in range(10):
            arr += 1


def section():
    arr = torch.zeros((1000, 1000))

    for i in range(N):
        with monit.section('run'):
            for t in range(10):
                arr += 1


def section_silent():
    arr = torch.zeros((1000, 1000))

    for i in range(N):
        with monit.section('run', is_silent=True):
            for t in range(10):
                arr += 1


def main():
    start = time.time()
    no_section()
    logger.log('No Section: ', (f'{time.time() - start}', Text.value))

    start = time.time()
    section()
    logger.log('Section: ', (f'{time.time() - start}', Text.value))

    start = time.time()
    section_silent()
    logger.log('Silent Section: ', (f'{time.time() - start}', Text.value))


if __name__ == '__main__':
    main()
