import tracemalloc

tracemalloc.start()


def main():
    import numpy as np
    from labml import monit, tracker, logger, lab
    with monit.section('Test'):
        print('test')


def wait():
    print('waiting')
    import time
    time.sleep(100)


def _snap():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:100]:
        print(stat)


if __name__ == '__main__':
    main()
    _snap()
    wait()
