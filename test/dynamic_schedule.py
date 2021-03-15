import time

from labml import experiment, monit, tracker
from labml.configs import BaseConfigs
from labml.internal.configs.schedule import FloatDynamicSchedule


class Configs(BaseConfigs):
    lr = FloatDynamicSchedule(0.01, (0, 1))


def main():
    experiment.create(name='test_schedule', writers={'screen', 'web_api'})
    lr = FloatDynamicSchedule(0.01, (0, 1))
    # experiment.configs({'lr': lr})
    conf = Configs()
    experiment.configs(conf)
    lr = conf.lr
    with experiment.start():
        for epoch in monit.loop(100):
            tracker.save('hp.lr', lr())
            time.sleep(1)


if __name__ == '__main__':
    main()
