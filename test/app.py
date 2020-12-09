from labml import tracker, experiment
from numpy.random import random


def main():
    conf = {'batch_size': 20}

    with experiment.record(name='sample', exp_conf=conf, writers={'web_api'}):
        for i in range(10000000):
            values = {'loss': random()}
            continue
            for j in range(0, 100):
                values[f'grad.fc.{j}.l1'] = random()
                values[f'grad.fc.{j}.l2'] = random()
                values[f'grad.fc.{j}.mean'] = random()

                # values[f'param.fc.{j}.l1'] = random()
                # values[f'param.fc.{j}.l2'] = random()
                # values[f'param.fc.{j}.mean'] = random()
                #
                # values[f'module.fc.{j}.l1'] = random()
                # values[f'module.fc.{j}.l2'] = random()
                # values[f'module.fc.{j}.mean'] = random()
                #
                # values[f'time.fc.{j}.l1'] = random()
                # values[f'time.fc.{j}.l2'] = random()
                # values[f'time.fc.{j}.mean'] = random()
            tracker.save(i, values)


if __name__ == '__main__':
    main()
