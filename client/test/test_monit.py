import time

from labml import monit, logger


def main():
    for _ in monit.loop(8):
        for t, idx in monit.mix(('train', range(10)), ('valid', range(2))):
            time.sleep(0.05)
        logger.log()
        with monit.section('Save', is_not_in_loop=True):
            time.sleep(0.1)


if __name__ == '__main__':
    main()