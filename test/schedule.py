import time

from labml.internal.configs.schedule import DynamicSchedule

from labml import experiment, monit, tracker


def main():
    experiment.create(name='test_schedule', writers={'screen', 'web_api'})
    lr = DynamicSchedule(0.01, (0, 1))
    experiment.configs({'lr': lr})
    with experiment.start():
        for epoch in monit.loop(100):
            tracker.save('hp.lr', lr())
            time.sleep(1)


if __name__ == '__main__':
    main()
