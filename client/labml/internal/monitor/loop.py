import math
import time
from typing import Optional, Dict, TYPE_CHECKING, List, Collection

from labml.internal.monitor.sections import LoopingSection
from labml.logger import Text
from ..tracker import tracker_singleton as tracker

if TYPE_CHECKING:
    from ..monitor import Monitor


class Loop:
    def __init__(self, iterator: Collection, *,
                 monitor: 'Monitor',
                 is_track: bool,
                 is_print_iteration_time: bool):
        """
        Creates an iterator with a range `iterator`.

        See example for usage.
        """
        self.iterator = iterator
        self._start_time = 0.
        self._iter_start_time = 0.
        self._iter_start_step = 0
        self._init_time = 0.
        self._iter_time = 0.
        self._beta_pow = 1.
        self._beta = 0.9
        self.monitor = monitor
        self.__global_step: Optional[int] = None
        self.__looping_sections: Dict[str, LoopingSection] = {}
        self._is_print_iteration_time = is_print_iteration_time
        self._is_track = is_track
        self._started = False

    def __iter__(self):
        self.iterator_iter = iter(self.iterator)
        self._start_time = time.time()
        self._init_time = 0.
        self._iter_time = 0.
        self._iter_start_step = tracker().global_step
        self._started = False
        self.monitor.start_loop()
        return self

    def __next__(self):
        try:
            next_value = next(self.iterator_iter)
        except StopIteration as e:
            self.monitor.finish_loop()
            raise e

        now = time.time()
        if not self._started:
            self.__init_time = now - self._start_time
        else:
            self._beta_pow *= self._beta
            self._iter_time *= self._beta
            self._iter_time += (1 - self._beta) * (now - self._iter_start_time)
            if self._is_track:
                tracker().store('time.loop',
                                (now - self._iter_start_time) /
                                max(1, tracker().global_step - self._iter_start_step))

        self._iter_start_time = now
        self._iter_start_step = tracker().global_step
        self._started = True

        tracker().loop_count(next_value)

        return next_value

    def log_progress(self):
        """
        Show progress
        """
        now = time.time()
        spent = now - self._start_time

        if not math.isclose(self._iter_time, 0.):
            estimate = self._iter_time / (1 - self._beta_pow)
        else:
            estimate = sum([s.get_estimated_time() for s in self.__looping_sections.values() if not s.is_child])

        total_time = estimate * len(self.iterator) + self._init_time
        total_time = max(total_time, spent)
        remain = total_time - spent

        remain /= 60
        spent /= 60
        estimate *= 1000

        spent_h = int(spent // 60)
        spent_m = int(spent % 60)
        remain_h = int(remain // 60)
        remain_m = int(remain % 60)

        to_print = [("  ", None)]
        if self._is_print_iteration_time:
            to_print.append((f"{estimate:,.0f}ms", Text.meta))
        to_print.append((f"{spent_h:3d}:{spent_m:02d}m/{remain_h:3d}:{remain_m:02d}m  ", Text.meta2))

        return to_print

    def get_section(self, *, name: str,
                    is_silent: bool,
                    is_timed: bool,
                    is_partial: bool,
                    total_steps: float,
                    parents: List[str]):
        key = '.'.join(parents + [name])
        if key not in self.__looping_sections:
            self.__looping_sections[key] = LoopingSection(monitor=self.monitor,
                                                          name=name,
                                                          is_silent=is_silent,
                                                          is_timed=is_timed,
                                                          is_track=self._is_track,
                                                          is_partial=is_partial,
                                                          total_steps=total_steps,
                                                          parents=parents)
        return self.__looping_sections[key]

    def log_sections(self):
        parts = []
        for name, section in self.__looping_sections.items():
            parts += section.log()

        return parts
