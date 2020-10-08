from numpy.random import random

from labml import tracker, experiment


def train(i):
    return 0.999 ** i + random() / 10, 1 - .999 ** i + random() / 10


conf = {'batch_size': 20}
with experiment.record(name='sample', exp_conf=conf, token='49d688f6624d468394ca029ecabc8a84'):
    for i in range(10000):
        loss, accuracy = train(i)
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
