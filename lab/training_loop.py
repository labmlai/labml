import signal
from typing import Any, Tuple, Optional

from . import logger
from .logger.colors import Text
from .configs import Configs


class TrainingLoopConfigs(Configs):
    loop_count: int = 10
    is_save_models: bool = False
    log_new_line_interval: int = 1
    log_write_interval: int = 1
    save_models_interval: int = 1

    training_loop: 'TrainingLoop'


@TrainingLoopConfigs.calc('training_loop')
def _loop_configs(c: TrainingLoopConfigs):
    return TrainingLoop(loop_count=c.loop_count,
                        is_save_models=c.is_save_models,
                        log_new_line_interval=c.log_new_line_interval,
                        log_write_interval=c.log_write_interval,
                        save_models_interval=c.save_models_interval)


class TrainingLoop:
    __signal_received: Optional[Tuple[Any, Any]]

    def __init__(self, *,
                 loop_count,
                 is_save_models,
                 log_new_line_interval,
                 log_write_interval,
                 save_models_interval):
        self.__loop_count = loop_count
        self.__is_save_models = is_save_models
        self.__log_new_line_interval = log_new_line_interval
        self.__log_write_interval = log_write_interval
        self.__save_models_interval = save_models_interval
        self.__loop = logger.loop(range(self.__loop_count))
        self.__signal_received = None

    def __iter__(self):
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
        logger.write()
        logger.new_line()
        if self.__is_save_models:
            logger.save_checkpoint()

    @staticmethod
    def __is_interval(epoch, interval):
        if epoch == 0:
            return False

        if epoch % interval == 0:
            return True
        else:
            return False

    def __next__(self):
        if self.__signal_received is not None:
            logger.log('\nKilling Loop.',
                       color=Text.danger)
            logger.finish_loop()
            self.__finish()
            raise StopIteration("SIGINT")

        try:
            epoch = next(self.__loop)
        except StopIteration as e:
            self.__finish()
            raise e

        if self.__is_interval(epoch, self.__log_write_interval):
            logger.write()
        if self.__is_interval(epoch, self.__log_new_line_interval):
            logger.new_line()

        if (self.__is_save_models and
                self.__is_interval(epoch, self.__save_models_interval)):
            logger.save_checkpoint()

        return epoch

    def __handler(self, sig, frame):
        # Pass second interrupt without delaying
        if self.__signal_received is not None:
            self.old_handler(*self.__signal_received)
            return

        # Store the interrupt signal for later
        self.__signal_received = (sig, frame)
        logger.log('\nSIGINT received. Delaying KeyboardInterrupt.',
                   color=Text.danger)
