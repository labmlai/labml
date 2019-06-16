import time

from lab import colors
from lab import logger


class MonitorIterator:
    """
    ### Monitor an iterator

    *Should be initialized only via `Logger`.*
    """

    def __init__(self, iterator: range, *,
                 logger: 'logger.Logger'):
        """
        Creates an iterator with a range `iterator`.

        See example for usage.
        """
        self.iterator = iterator
        self.sections = {}
        self._start_time = 0
        self.steps = len(iterator)
        self.counter = 0
        self.logger = logger

    def section(self, name: str, is_monitored=True):
        """
        Creates a monitored section with given name.
        """

        if is_monitored:
            if name not in self.sections:
                self.sections[name] = _MonitorIteratorSection(self, name,
                                                              logger=self.logger)
        else:
            if name not in self.sections:
                self.sections[name] = _UnmonitoredIteratorSection(self, name,
                                                                  logger=self.logger)

        return self.sections[name]

    def unmonitored(self, name: str):
        """
        Creates an unmonitored section with given name.
        """

        return self.section(name, is_monitored=False)

    def __iter__(self):
        self.iterator_iter = iter(self.iterator)
        self._start_time = time.time()
        self.counter = 0
        return self

    def __next__(self):
        try:
            next_value = next(self.iterator_iter)
        except StopIteration as e:
            raise e

        self.counter += 1

        return next_value

    def progress(self):
        """
        Show progress
        """
        spent = (time.time() - self._start_time) / 60
        remain = self.steps * spent / self.counter - spent

        spent_h = int(spent // 60)
        spent_m = int(spent % 60)
        remain_h = int(remain // 60)
        remain_m = int(remain % 60)

        self.logger.log(f"  {spent_h:3d}:{spent_m:02d}m/{remain_h:3d}:{remain_m:02d}m  ",
                        color=colors.BrightColor.purple,
                        new_line=False)


class _MonitorIteratorSection:
    """
    ### Monitors a section of code within an iterator

    *Should only be initialzed via `_MonitorIterator`.*

    It keeps track of moving exponentiol average of
     time spent on the section through out all iterations.
    """

    def __init__(self, parent, name: str, *,
                 logger: 'logger.Logger'):
        self.parent = parent
        self.name = name
        self._start_time = 0

        self.beta = 0.9
        self.beta_pow = 1

        self.estimated_time = 0

        self.logger = logger

    def get_time(self):
        """
        Get the moving exponential average time.
        """
        if self.estimated_time <= 0:
            return self.estimated_time

        return self.estimated_time / (1 - self.beta_pow)

    def add_time(self, elapsed):
        """
        Update moving average time
        """
        self.beta_pow *= self.beta
        self.estimated_time = self.beta * self.estimated_time + (1 - self.beta) * elapsed

    def __enter__(self):
        self.logger.log(f"{self.name}:", new_line=False)
        self.logger.log(" ...", color=colors.Color.orange, new_line=False)
        self.logger.log(" " * 8, new_line=False)
        self.logger.pop_current_line()
        self._start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.add_time(time.time() - self._start_time)
        self.logger.pop_current_line()
        time_ms = 1000 * self.get_time()
        self.logger.log(f"{time_ms:10,.2f}ms  ",
                        color=colors.BrightColor.cyan,
                        new_line=False)


class _UnmonitoredIteratorSection:
    """
    ### Unmonitored section in a monitored iterator.

    Used for structuring code.
    """

    def __init__(self, parent, name: str, *,
                 logger: 'logger.Logger'):
        self.parent = parent
        self.name = name
        self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
