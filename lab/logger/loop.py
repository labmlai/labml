import time
from typing import Optional, Dict

from lab import colors
from lab.logger.sections import Section, section_factory
from lab import logger as logger_base


class Loop:
    def __init__(self, iterator: range, *,
                 logger: 'logger_base.Logger'):
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
        self.__global_step: Optional[int] = None
        self.__looping_sections: Dict[str, Section] = {}

    @property
    def global_step(self):
        if self.__global_step is not None:
            return self.__global_step

        return self.counter

    @global_step.setter
    def global_step(self, value):
        self.__global_step = value

    def __iter__(self):
        self.iterator_iter = iter(self.iterator)
        self._start_time = time.time()
        self.counter = 0
        return self

    def __next__(self):
        try:
            next_value = next(self.iterator_iter)
        except StopIteration as e:
            self.logger.finish_loop()
            raise e

        self.counter += 1

        return next_value

    def log_progress(self):
        """
        Show progress
        """
        spent = (time.time() - self._start_time) / 60
        remain = self.steps * spent / self.counter - spent

        spent_h = int(spent // 60)
        spent_m = int(spent % 60)
        remain_h = int(remain // 60)
        remain_m = int(remain % 60)

        return [(f"  {spent_h:3d}:{spent_m:02d}m/{remain_h:3d}:{remain_m:02d}m  ",
                 colors.BrightColor.purple)]

    def log_global_step(self):
        return [(f"{self.global_step :8,}:  ",
                 colors.BrightColor.orange)]

    def get_section(self, *, name: str,
                    is_silent: bool,
                    is_timed: bool,
                    is_partial: bool,
                    total_steps: float):
        if name not in self.__looping_sections:
            self.__looping_sections[name] = section_factory(logger=self.logger,
                                                            name=name,
                                                            is_silent=is_silent,
                                                            is_timed=is_timed,
                                                            is_partial=is_partial,
                                                            total_steps=total_steps,
                                                            is_looping=True)
        return self.__looping_sections[name]

    def log_sections(self):
        parts = []
        for name, section in self.__looping_sections.items():
            parts += section.log()

        return parts
