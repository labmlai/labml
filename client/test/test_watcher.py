import time

from labml import experiment
from labml_helpers.training_loop import TrainingLoop


def main():
    experiment.create(name='watcher')
    loop = TrainingLoop(loop_count=10, loop_step=1, is_save_models=False, log_new_line_interval=1,
                        log_write_interval=1, save_models_interval=1, is_loop_on_interrupt=False)
    with experiment.start():
        for i in loop:
            # time.sleep(10)
            print(i)
            # if i == 5:
            #     raise RuntimeError('oops')


if __name__ == '__main__':
    main()
