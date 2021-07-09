import time

from numpy.random import random

from labml import tracker, experiment

conf = {'batch_size': 20}


def train(n: int):
    return 0.999 ** n + random() / 10, 1 - .999 ** n + random() / 10


with experiment.record(name='sample', exp_conf=conf):
    for i in range(100000):
        time.sleep(0.2)
        loss, accuracy = train(i)
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
