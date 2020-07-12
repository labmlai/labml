from pathlib import PurePath

from labml import manage


def _test():
    manage.new_run(PurePath('/home/varuna/ml/lab/test/mnist.py'),
                   {'optimizer': 'sgd_optimizer'})


def _test_process():
    manage.new_run_process(PurePath('/home/varuna/ml/lab/test/mnist.py'),
                           {'optimizer': 'sgd_optimizer'})
    # manage.new_run_process(PurePath('/home/varuna/ml/lab/test/mnist.py'),
    #                        {'optimizer': 'adam_optimizer'})


if __name__ == '__main__':
    # _test()
    _test_process()
