from numpy.random import random

from labml import tracker, experiment


def train(i):
    return 0.999 ** i + random() / 10, 1 - .999 ** i + random() / 10


conf = {'batch_size': 20}
with experiment.record(name='sample', exp_conf=conf, token='903c84fba8ca49ca9f215922833e08cf'):
    for i in range(10000):
        loss, accuracy = train(i)
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
