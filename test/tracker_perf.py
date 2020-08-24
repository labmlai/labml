import time

import torch

from labml import experiment, monit, tracker, logger
from labml.logger import Text

N = 10_000


def no_tracking():
    arr = torch.zeros((1000, 1000))

    for i in range(N):
        for t in range(10):
            arr += 1


def setup_and_add_save():
    arr = torch.zeros((1000, 1000))
    for t in range(10):
        tracker.set_scalar(f"loss1.{t}", is_print=t == 0)

    experiment.start()

    for i in monit.loop(N):
        for t in range(10):
            arr += 1
        for t in range(10):
            tracker.add({f'loss1.{t}': i})
            tracker.save()


def add_save():
    arr = torch.zeros((1000, 1000))
    experiment.start()

    for i in monit.loop(N):
        for t in range(10):
            arr += 1
        for t in range(10):
            if i == 0:
                tracker.set_scalar(f"loss1.{t}", is_print=t == 0)
        for t in range(10):
            tracker.add({f'loss1.{t}': i})
            tracker.save()


def add_only():
    arr = torch.zeros((1000, 1000))
    experiment.start()

    for i in monit.loop(N):
        for t in range(10):
            arr += 1
        for t in range(10):
            if i == 0:
                tracker.set_scalar(f"loss1.{t}", is_print=t == 0)
        for t in range(10):
            tracker.add({f'loss1.{t}': i})

    tracker.save()


def main():
    experiment.create(writers={'sqlite'})

    start = time.time()
    no_tracking()
    logger.log('Time taken without tracking: ', (f'{time.time() - start}', Text.value))

    start = time.time()
    setup_and_add_save()
    logger.log('Time taken setup and add: ', (f'{time.time() - start}', Text.value))

    start = time.time()
    add_save()
    logger.log('Time taken add: ', (f'{time.time() - start}', Text.value))

    start = time.time()
    add_only()
    logger.log('Time taken add only: ', (f'{time.time() - start}', Text.value))


if __name__ == '__main__':
    main()
