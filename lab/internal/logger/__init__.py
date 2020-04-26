import typing
from pathlib import PurePath
from typing import Optional, List, Union, Tuple, Dict

from .artifacts import Artifact
from .colors import StyleCode
from .delayed_keyboard_interrupt import DelayedKeyboardInterrupt
from .destinations.factory import create_destination
from .indicators import Indicator
from .iterator import Iterator
from .loop import Loop
from .sections import Section, OuterSection
from .store import Store
from .writers import Writer, ScreenWriter
from ...logger import Text


class Logger:
    def __init__(self):
        self.__store = Store(self)
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

    def log(self, parts: List[Union[str, Tuple[str, StyleCode]]], *,
            is_new_line=True):
        self.__destination.log(parts, is_new_line=is_new_line)

    def add_indicator(self, indicator: Indicator):
        self.__store.add_indicator(indicator)

    def add_artifact(self, artifact: Artifact):
        self.__store.add_artifact(artifact)

    def save_indicators(self, file: PurePath):
        self.__store.save_indicators(file)

    def save_artifacts(self, file: PurePath):
        self.__store.save_artifactors(file)

    def store(self, *args, **kwargs):
        self.__store.store(*args, **kwargs)

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

    def new_line(self):
        self.__destination.new_line()

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
                total_steps: Optional[int] = None, *,
                is_silent: bool = False,
                is_timed: bool = True):
        return Iterator(logger=self,
                        name=name,
                        iterable=iterable,
                        is_silent=is_silent,
                        is_timed=is_timed,
                        total_steps=total_steps,
                        is_enumerate=False)

    def enum(self, name, iterable: typing.Sized, *,
             is_silent: bool = False,
             is_timed: bool = True):
        return Iterator(logger=self,
                        name=name,
                        iterable=iterable,
                        is_silent=is_silent,
                        is_timed=is_timed,
                        total_steps=None,
                        is_enumerate=True)

    def section(self, name, *,
                is_silent: bool = False,
                is_timed: bool = True,
                is_partial: bool = False,
                is_new_line: bool = True,
                total_steps: float = 1.0):

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
             is_print_iteration_time=True):
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

    def section_enter(self, section):
        if len(self.__sections) == 0:
            raise RuntimeError("Entering a section without creating a section.\n"
                               "Always use logger.section to create a section")

        if section is not self.__sections[-1]:
            raise RuntimeError("Entering a section other than the one last_created\n"
                               "Always user with logger.section(...):")

        if len(self.__sections) > 1 and not self.__sections[-2].is_parented:
            self.__sections[-2].make_parent()
            self.new_line()

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

        self.log(self.__sections[-1].log(), is_new_line=False)

    def section_exit(self, section):
        if len(self.__sections) == 0:
            raise RuntimeError("Impossible")

        if section is not self.__sections[-1]:
            raise RuntimeError("Impossible")

        self.__log_line()
        self.__sections.pop(-1)

    def delayed_keyboard_interrupt(self):
        return DelayedKeyboardInterrupt(self)

    def _log_key_value(self, items: List[Tuple[any, any]], is_show_count=True):
        max_key_len = 0
        for k, v in items:
            max_key_len = max(max_key_len, len(str(k)))

        count = 0
        for k, v in items:
            count += 1
            spaces = " " * (max_key_len - len(str(k)))
            s = str(v)
            if len(s) > 80:
                s = f"{s[:80]} ..."
            self.log([(f"{spaces}{k}: ", Text.key),
                      (s, Text.value)])

        if is_show_count:
            self.log([
                "Total ",
                (str(count), Text.meta),
                " item(s)"])

    def info(self, *args, **kwargs):
        if len(args) == 0:
            self._log_key_value([(k, v) for k, v in kwargs.items()], False)
        elif len(args) == 1:
            assert len(kwargs.keys()) == 0
            arg = args[0]
            if type(arg) == list:
                self._log_key_value([(i, v) for i, v in enumerate(arg)])
            elif type(arg) == dict:
                self._log_key_value([(k, v) for k, v in arg.items()])
        else:
            assert len(kwargs.keys()) == 0
            self._log_key_value([(i, v) for i, v in enumerate(args)], False)


_internal: Optional[Logger] = None


def logger_singleton() -> Logger:
    global _internal
    if _internal is None:
        _internal = Logger()

    return _internal
