import time

from labml import experiment, monit, tracker
from labml.configs import BaseConfigs
from labml.configs import FloatDynamicHyperParam


class Configs(BaseConfigs):
    lr = FloatDynamicHyperParam(0.01, (0, 1))


def main():
    experiment.create(name='test_dynamic_hp', writers={'screen', 'web_api'})
    lr = FloatDynamicHyperParam(0.01, (0, 1))
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
