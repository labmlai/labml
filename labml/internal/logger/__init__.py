import typing
from pathlib import PurePath
from typing import Optional, List, Union, Tuple, Dict

from .destinations.factory import create_destination
from .inspect import Inspect
from .iterator import Iterator
from .loop import Loop
from .sections import Section, OuterSection
from .store import Store
from .store.indicators import Indicator
from .writers import Writer
from .writers.screen import ScreenWriter
from ..util.colors import StyleCode
from ...logger import Text


class Logger:
    """
    This handles the interactions among sections, loop and store
    """

    def __init__(self):
        self.__store = Store()
        self.__writers: List[Writer] = []

        self.__loop: Optional[Loop] = None
        self.__sections: List[Section] = []

        self.__indicators_print = []

        self.__screen_writer = ScreenWriter()
        self.__writers.append(self.__screen_writer)

        self.__start_global_step: Optional[int] = None
        self.__global_step: Optional[int] = None
        self.__last_global_step: Optional[int] = None

        self.__destination = create_destination()
        self.__inspect = Inspect(self)

    @property
    def global_step(self) -> int:
        if self.__global_step is not None:
            return self.__global_step

        global_step = 0
        if self.__start_global_step is not None:
            global_step = self.__start_global_step

        if self.__is_looping():
            return global_step + self.__loop.counter

        if self.__last_global_step is not None:
            return self.__last_global_step

        return global_step

    def add_writer(self, writer: Writer):
        self.__writers.append(writer)

    def reset_writers(self):
        self.__writers = []
        self.__writers.append(self.__screen_writer)

    def log(self, parts: List[Union[str, Tuple[str, Optional[StyleCode]]]], *,
            is_new_line=True):
        self.__destination.log(parts, is_new_line=is_new_line)

    def reset_store(self):
        self.__store = Store()

    def add_indicator(self, indicator: Indicator):
        self.__store.add_indicator(indicator)

    def save_indicators(self, file: PurePath):
        self.__store.save_indicators(file)

    def store(self, key: str, value: any):
        self.__store.store(key, value)

    def store_namespace(self, name: str):
        return self.__store.create_namespace(name)

    def set_global_step(self, global_step: Optional[int]):
        self.__global_step = global_step

    def set_start_global_step(self, global_step: Optional[int]):
        self.__start_global_step = global_step

    def add_global_step(self, increment_global_step: int = 1):
        if self.__global_step is None:
            if self.__start_global_step is not None:
                self.__global_step = self.__start_global_step
            else:
                self.__global_step = 0

        self.__global_step += increment_global_step

    def __is_looping(self):
        if self.__loop is not None and self.__loop.is_started:
            return True
        else:
            return False

    def write_h_parameters(self, hparams: Dict[str, any]):
        for w in self.__writers:
            w.write_h_parameters(hparams)

    def write(self):
        global_step = self.global_step

        for w in self.__writers:
            if w != self.__screen_writer:
                self.__store.write(w, global_step)
        self.__indicators_print = self.__store.write(self.__screen_writer, global_step)
        self.__store.clear()

        parts = [(f"{self.global_step :8,}:  ", Text.highlight)]
        if self.__is_looping():
            self.__log_looping_line()
        else:
            parts += self.__indicators_print
            self.log(parts, is_new_line=False)

    def iterate(self, name, iterable: Union[typing.Iterable, typing.Sized, int],
                total_steps: Optional[int], *,
                is_silent: bool,
                is_timed: bool):
        return Iterator(logger=self,
                        name=name,
                        iterable=iterable,
                        is_silent=is_silent,
                        is_timed=is_timed,
                        total_steps=total_steps,
                        is_enumerate=False)

    def enum(self, name, iterable: typing.Sized, *,
             is_silent: bool,
             is_timed: bool):
        return Iterator(logger=self,
                        name=name,
                        iterable=iterable,
                        is_silent=is_silent,
                        is_timed=is_timed,
                        total_steps=None,
                        is_enumerate=True)

    def section(self, name, *,
                is_silent: bool,
                is_timed: bool,
                is_partial: bool,
                is_new_line: bool,
                total_steps: float):

        if self.__is_looping():
            if len(self.__sections) != 0:
                raise RuntimeError("No nested sections within loop")

            section = self.__loop.get_section(name=name,
                                              is_silent=is_silent,
                                              is_timed=is_timed,
                                              is_partial=is_partial,
                                              total_steps=total_steps)
            self.__sections.append(section)
        else:
            self.__sections.append(OuterSection(logger=self,
                                                name=name,
                                                is_silent=is_silent,
                                                is_timed=is_timed,
                                                is_partial=is_partial,
                                                is_new_line=is_new_line,
                                                total_steps=total_steps,
                                                level=len(self.__sections)))

        return self.__sections[-1]

    def progress(self, steps: float):
        if len(self.__sections) == 0:
            raise RuntimeError("You must be within a section to report progress")

        if self.__sections[-1].progress(steps):
            self.__log_line()

    def set_successful(self, is_successful=True):
        if len(self.__sections) == 0:
            raise RuntimeError("You must be within a section to report success")

        self.__sections[-1].is_successful = is_successful
        self.__log_line()

    def loop(self, iterator_: range, *,
             is_print_iteration_time: bool):
        if len(self.__sections) != 0:
            raise RuntimeError("Cannot start a loop within a section")

        self.__loop = Loop(iterator=iterator_, logger=self,
                           is_print_iteration_time=is_print_iteration_time)
        return self.__loop

    def finish_loop(self):
        if len(self.__sections) != 0:
            raise RuntimeError("Cannot be within a section when finishing the loop")
        self.__last_global_step = self.global_step
        self.__loop = None
        for w in self.__writers:
            w.flush()

    def section_enter(self, section):
        if len(self.__sections) == 0:
            raise RuntimeError("Entering a section without creating a section.\n"
                               "Always use logger.section to create a section")

        if section is not self.__sections[-1]:
            raise RuntimeError("Entering a section other than the one last_created\n"
                               "Always user with logger.section(...):")

        if len(self.__sections) > 1 and not self.__sections[-2].is_parented:
            self.__sections[-2].make_parent()
            if not self.__sections[-1].is_silent:
                self.log([])

        self.__log_line()

    def __log_looping_line(self):
        parts = [(f"{self.global_step :8,}:  ", Text.highlight)]
        parts += self.__loop.log_sections()
        parts += self.__indicators_print
        parts += self.__loop.log_progress()

        self.log(parts, is_new_line=False)

    def __log_line(self):
        if self.__is_looping():
            self.__log_looping_line()
            return

        if len(self.__sections) == 0:
            return

        parts = self.__sections[-1].log()
        if parts is None:
            return

        self.log(parts, is_new_line=False)

    def section_exit(self, section):
        if len(self.__sections) == 0:
            raise RuntimeError("Impossible")

        if section is not self.__sections[-1]:
            raise RuntimeError("Impossible")

        self.__log_line()
        self.__sections.pop(-1)

    def info(self, *args, **kwargs):
        self.__inspect.info(*args, **kwargs)


_internal: Optional[Logger] = None


def logger_singleton() -> Logger:
    global _internal
    if _internal is None:
        _internal = Logger()

    return _internal
