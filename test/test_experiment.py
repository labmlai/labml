import time

from labml import tracker, experiment


def main():
    experiment.create(name='Test')

    with experiment.start():
        for i in range(1, 401):
            tracker.add_global_step()
            time.sleep(1)
            tracker.save(loss=1.)


if __name__ == '__main__':
    main()
