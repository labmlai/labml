from labml import tracker, monit, experiment, logger
from labml.configs import BaseConfigs


class Configs(BaseConfigs):
    epochs: int = 5

    def run(self):
        tracker.set_text('text_artifact', is_print=True)
        tracker.set_indexed_text('ti', is_print=True)
        tracker.set_indexed_text('other', is_print=True)
        for i in monit.loop(self.epochs):
            tracker.add('text_artifact', f'sample {i}')
            for j in range(5):
                tracker.add('ti', (f'{j}', 'text' * 5 + f'text {i} {j}'))
                tracker.add('other', (f'{j}', f'other {j}'))

            tracker.save()
            logger.log()


def main():
    conf = Configs()
    experiment.create(name='test_artifacts', writers={'sqlite'})
    experiment.configs(conf, 'run')
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
