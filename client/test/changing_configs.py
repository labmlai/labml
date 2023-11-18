import time

from numpy.random import random

from labml import tracker, experiment


def main():
    conf = {'batch_size': 20}

    with experiment.record(name='sample', exp_conf=conf, writers={'app', 'screen'}):
        for i in range(10_000):
            values = {'loss': random()}
            tracker.save(i, values)

            time.sleep(.01)

            if i % 1000 == 0:
                experiment.configs({f'conf_{i}': f'my_value_{i}'})
                tracker.new_line()


if __name__ == '__main__':
    main()
