import signal
from typing import Optional, Tuple, Any, Collection

import typing

from labml import tracker, logger, experiment, monit
from labml.configs import BaseConfigs, meta_config, option
from labml.internal.monitor import Loop
from labml.logger import Text


class TrainingLoopIterator(Collection):
    def __init__(self, start: int, total: int, step: Optional[int]):
        self.step = step
        self.total = total
        self.start = start
        self.i = None

    def __iter__(self):
        self.i = None
        return self

    def __next__(self):
        if self.step is not None:
            if self.i is None:
                self.i = self.start
            else:
                self.i += self.step
        else:
            if self.i is None:
                self.i = 0
            else:
                self.i += 1

        if self.i >= self.total:
            raise StopIteration()

        if self.step is None:
            return tracker.get_global_step()
        else:
            return self.i

    def __len__(self) -> int:
        if self.step is not None:
            return (self.total - self.start) // self.step
        else:
            return self.total

    def __contains__(self, x: object) -> bool:
        return False


class TrainingLoop:
    __loop: Loop
    __signal_received: Optional[Tuple[Any, Any]]

    def __init__(self, *,
                 loop_count: int,
                 loop_step: Optional[int],
                 is_save_models: bool,
                 log_new_line_interval: int,
                 log_write_interval: int,
                 save_models_interval: int,
                 is_loop_on_interrupt: bool):
        self.__loop_count = loop_count
        self.__loop_step = loop_step
        self.__is_save_models = is_save_models
        self.__log_new_line_interval = log_new_line_interval
        self.__log_write_interval = log_write_interval
        self.__last_write_step = 0
        self.__last_new_line_step = 0
        self.__save_models_interval = save_models_interval
        self.__last_save_step = 0
        self.__signal_received = None
        self.__is_loop_on_interrupt = is_loop_on_interrupt
        self._iter = None

    def __iter__(self):
        self._iter = TrainingLoopIterator(tracker.get_global_step(),
                                          self.__loop_count,
                                          self.__loop_step)

        self.__loop = monit.loop(typing.cast(Collection, self._iter))

        iter(self.__loop)
        try:
            self.old_handler = signal.signal(signal.SIGINT, self.__handler)
        except ValueError:
            pass
        return self

    def __finish(self):
        try:
            signal.signal(signal.SIGINT, self.old_handler)
        except ValueError:
            pass
        tracker.save()
        tracker.new_line()
        if self.__is_save_models:
            logger.log("Saving model...")
            experiment.save_checkpoint()

    # def is_interval(self, interval: int, global_step: Optional[int] = None):
    #     if global_step is None:
    #         global_step = tracker.get_global_step()
    #
    #     if global_step - self.__loop_step < 0:
    #         return False
    #
    #     if global_step // interval > (global_step - self.__loop_step) // interval:
    #         return True
    #     else:
    #         return False

    def __next__(self):
        if self.__signal_received is not None:
            logger.log('\nKilling Loop.', Text.danger)
            monit.finish_loop()
            self.__finish()
            raise StopIteration("SIGINT")

        try:
            global_step = next(self.__loop)
        except StopIteration as e:
            self.__finish()
            raise e

        tracker.set_global_step(global_step)

        if global_step - self.__last_write_step >= self.__log_write_interval:
            tracker.save()
            self.__last_write_step = global_step
        if global_step - self.__last_new_line_step >= self.__log_new_line_interval:
            tracker.new_line()
            self.__last_new_line_step = global_step
        # if self.is_interval(self.__log_write_interval, global_step):
        #     tracker.save()
        # if self.is_interval(self.__log_new_line_interval, global_step):
        #     logger.log()

        # if (self.__is_save_models and
        #         self.is_interval(self.__save_models_interval, global_step)):
        #     experiment.save_checkpoint()
        if (self.__is_save_models and
                global_step - self.__last_save_step >= self.__save_models_interval):
            experiment.save_checkpoint()

        return global_step

    def __handler(self, sig, frame):
        # Pass second interrupt without delaying
        if self.__signal_received is not None:
            logger.log('\nSIGINT received twice. Stopping...', Text.danger)
            self.old_handler(*self.__signal_received)
            return

        if self.__is_loop_on_interrupt:
            # Store the interrupt signal for later
            self.__signal_received = (sig, frame)
            logger.log('\nSIGINT received. Delaying KeyboardInterrupt.', Text.danger)
        else:
            self.__finish()
            logger.log('Killing loop...', Text.danger)
            self.old_handler(sig, frame)

    def __str__(self):
        return "LabTrainingLoop"


class TrainingLoopConfigs(BaseConfigs):
    loop_count: int = 10
    loop_step: int = 1
    is_save_models: bool = False
    log_new_line_interval: int = 1
    log_write_interval: int = 1
    save_models_interval: int = 1
    is_loop_on_interrupt: bool = False

    training_loop: TrainingLoop


@option(TrainingLoopConfigs.training_loop)
def _loop_configs(c: TrainingLoopConfigs):
    return TrainingLoop(loop_count=c.loop_count,
                        loop_step=c.loop_step,
                        is_save_models=c.is_save_models,
                        log_new_line_interval=c.log_new_line_interval,
                        log_write_interval=c.log_write_interval,
                        save_models_interval=c.save_models_interval,
                        is_loop_on_interrupt=c.is_loop_on_interrupt)


meta_config(TrainingLoopConfigs.loop_step,
            TrainingLoopConfigs.loop_count,
            TrainingLoopConfigs.is_save_models,
            TrainingLoopConfigs.log_new_line_interval,
            TrainingLoopConfigs.log_write_interval,
            TrainingLoopConfigs.save_models_interval,
            TrainingLoopConfigs.is_loop_on_interrupt)
