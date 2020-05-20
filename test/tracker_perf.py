import time

from labml import experiment, monit, tracker, logger
from labml.logger import Text


def setup_and_add():
    for t in range(10):
        tracker.set_scalar(f"loss1.{t}", is_print=t == 0)

    experiment.start()

    for i in monit.loop(1000):
        for t in range(10):
            tracker.add({f'loss1.{t}': i})
            tracker.save()


def add():
    experiment.start()

    for i in monit.loop(1000):
        for t in range(10):
            if i == 0:
                tracker.set_scalar(f"loss1.{t}", is_print=t == 0)
        for t in range(10):
            tracker.add({f'loss1.{t}': i})
            tracker.save()


def main():
    experiment.create(writers={'sqlite'})

    start = time.time()
    setup_and_add()
    logger.log('Time taken: ', (f'{time.time() - start}', Text.value))


if __name__ == '__main__':
    main()
