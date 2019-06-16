import time

from lab import colors
from lab import logger


class Monitor:
    """
    ### Monitors a section of code

    *Should be initialized via `Logger`.*

    This is used monitor time taken for the section of code to run.
    It also helps structure the code.

    You can set `is_successful` to `False` if the execution failed.
    """

    def __init__(self, name: str, *,
                 logger: 'logger.Logger',
                 silent: bool = False, timed: bool = True):
        """

        :param name: name of the section of code
        :param silent: whether or not to output to screen
        :param timed: whether to time the section of code
        """
        self.name = name
        self.silent = silent
        self.timed = timed
        self._start_time = 0
        self.is_successful = True
        self.logger = logger

    def __enter__(self):
        if self.silent:
            return self

        self.logger.log(f"{self.name}...", new_line=False)

        if self.timed:
            self._start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.silent:
            return

        parts = []
        if self.is_successful:
            parts.append(("[DONE]", colors.BrightColor.green))
        else:
            parts.append(("[FAIL]", colors.BrightColor.red))

        if self.timed:
            time_end = time.time()
            duration_ms = 1000 * (time_end - self._start_time)
            parts.append((f"\t{duration_ms :,.2f}ms",
                          colors.BrightColor.cyan))

        self.logger.log_color(parts)
