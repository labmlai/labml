import signal
from typing import Optional, Tuple, Any

from lab import loop, tracker, logger, experiment
from lab.internal.logger import Loop
from lab.logger import Text


class TrainingLoop:
    __loop: Loop
    __signal_received: Optional[Tuple[Any, Any]]

    def __init__(self, *,
                 loop_count: int,
                 loop_step: int,
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
        self.__save_models_interval = save_models_interval
        self.__signal_received = None
        self.__is_loop_on_interrupt = is_loop_on_interrupt

    def __iter__(self):
        self.__loop = loop.loop(range(loop.get_global_step(),
                                      self.__loop_count,
                                      self.__loop_step))
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
        logger.log()
        if self.__is_save_models:
            logger.log("Saving model...")
            experiment.save_checkpoint()

    def __is_interval(self, global_step, interval):
        if global_step - self.__loop_step < 0:
            return False

        if global_step // interval > (global_step - self.__loop_step) // interval:
            return True
        else:
            return False

    def __next__(self):
        if self.__signal_received is not None:
            logger.log('\nKilling Loop.',
                       color=Text.danger)
            loop.finish_loop()
            self.__finish()
            raise StopIteration("SIGINT")

        try:
            global_step = next(self.__loop)
        except StopIteration as e:
            self.__finish()
            raise e

        loop.set_global_step(global_step)

        if self.__is_interval(global_step, self.__log_write_interval):
            tracker.save()
        if self.__is_interval(global_step, self.__log_new_line_interval):
            logger.log()

        if (self.__is_save_models and
                self.__is_interval(global_step, self.__save_models_interval)):
            experiment.save_checkpoint()

        return global_step

    def __handler(self, sig, frame):
        # Pass second interrupt without delaying
        if self.__signal_received is not None:
            logger.log('\nSIGINT received twice. Stopping...',
                       color=Text.danger)
            self.old_handler(*self.__signal_received)
            return

        if self.__is_loop_on_interrupt:
            # Store the interrupt signal for later
            self.__signal_received = (sig, frame)
            logger.log('\nSIGINT received. Delaying KeyboardInterrupt.',
                       color=Text.danger)
        else:
            self.__finish()
            logger.log('Killing loop...', Text.danger)
            self.old_handler(sig, frame)

    def __str__(self):
        return "LabTrainingLoop"