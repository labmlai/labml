from pathlib import PurePath

from labml import manage


def _test():
    manage.new_run(PurePath('/home/varuna/ml/lab/test/mnist.py'),
                   {'optimizer': 'sgd_optimizer'})


if __name__ == '__main__':
    _test()
