import time

from labml import experiment, monit, tracker


def main():
    experiment.create(name='tracker', writers={'screen', 'file'})

    with experiment.start():
        for i in monit.loop(50):
            for j in range(10):
                tracker.add('balance', i * 10 + j)
                time.sleep(0.1)
            tracker.save()


if __name__ == '__main__':
    main()
