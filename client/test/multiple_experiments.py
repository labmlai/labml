from labml import tracker, experiment
from numpy.random import random


def main():
    conf = {'batch_size': 20}

    for i in range(2):
        with experiment.record(name=f'sample_{i}', exp_conf=conf, writers={'screen'}):
            for epoch in range(100):
                tracker.save(i, loss=random())
            tracker.new_line()


if __name__ == '__main__':
    main()
