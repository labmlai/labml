import typing
from typing import Optional, List, Union, Tuple

from labml.internal.util.colors import StyleCode
from .iterator import Iterator
from .loop import Loop
from .mix import Mix
from .sections import Section, OuterSection
from ..logger import logger_singleton as logger
from ..logger.types import LogPart
from ..tracker import tracker_singleton as tracker
from ...logger import Text
from ...utils.notice import labml_notice


class Monitor:
    __loop_indicators: List[Union[str, Tuple[str, Optional[StyleCode]]]]
    __is_looping: bool

    def __init__(self):
        self.__loop: Optional[Loop] = None
        self.__sections: List[Section] = []
        self.__is_looping = False
        self.__loop_indicators = []
        self.__is_silent = False

    def clear(self):
        self.__loop: Optional[Loop] = None
        self.__sections: List[Section] = []
        self.__is_looping = False
        self.__loop_indicators = []

    def silent(self, is_silent: bool = True):
        self.__is_silent = is_silent

    def mix(self, total_iterations, iterators: List[Tuple[str, typing.Sized]],
            is_monit: bool):
        return Mix(total_iterations=total_iterations,
                   iterators=iterators,
                   is_monit=is_monit, logger=self)

    def iterate(self, name, iterable: Union[typing.Iterable, typing.Sized, int],
                total_steps: Optional[int], *,
                is_silent: bool,
                is_children_silent: bool,
                is_timed: bool,
                section: Optional[Section]):
        return Iterator(logger=self,
                        name=name,
                        iterable=iterable,
                        is_silent=is_silent,
                        is_timed=is_timed,
                        total_steps=total_steps,
                        is_children_silent=is_children_silent,
                        is_enumerate=False,
                        section=section)

    def enum(self, name, iterable: typing.Sized, *,
             is_silent: bool,
             is_children_silent: bool,
             is_timed: bool,
             section: Optional[Section]):
        return Iterator(logger=self,
                        name=name,
                        iterable=iterable,
                        is_silent=is_silent,
                        is_timed=is_timed,
                        total_steps=None,
                        is_children_silent=is_children_silent,
                        is_enumerate=True,
                        section=section)

    def section(self, name, *,
                is_silent: bool,
                is_timed: bool,
                is_partial: bool,
                is_new_line: bool,
                is_children_silent: bool,
                total_steps: float) -> Section:
        if self.__is_looping:
            if len(self.__sections) != 0:
                is_silent = True

            section = self.__loop.get_section(name=name,
                                              is_silent=is_silent,
                                              is_timed=is_timed,
                                              is_partial=is_partial,
                                              total_steps=total_steps,
                                              parents=[s.name for s in self.__sections])
            self.__sections.append(section)
        else:
            if len(self.__sections) > 0:
                if self.__sections[-1].is_silent or self.__sections[-1].is_children_silent:
                    is_silent = True
                    is_children_silent = True
            self.__sections.append(OuterSection(monitor=self,
                                                name=name,
                                                is_silent=is_silent,
                                                is_timed=is_timed,
                                                is_partial=is_partial,
                                                is_new_line=is_new_line,
                                                is_children_silent=is_children_silent,
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

    def loop(self, iterator_: typing.Collection, *,
             is_track: bool,
             is_print_iteration_time: bool):
        if len(self.__sections) != 0:
            labml_notice(['LabML Loop: ', ('Starting loop inside sections', Text.key), '\n',
                          (
                              'This could be because some iterators crashed in a previous cell in a notebook.',
                              Text.meta)],
                         is_danger=False)
            err = RuntimeError('Section outside loop')
            for s in reversed(self.__sections):
                s.__exit__(type(err), err, err.__traceback__)
            # raise RuntimeError("Cannot start a loop within a section")

        self.__loop = Loop(iterator=iterator_,
                           monitor=self,
                           is_track=is_track,
                           is_print_iteration_time=is_print_iteration_time)
        return self.__loop

    def start_loop(self):
        self.__is_looping = True
        tracker().start_loop(self.set_looping_indicators)

    def finish_loop(self):
        if len(self.__sections) != 0:
            raise RuntimeError("Cannot be within a section when finishing the loop")
        tracker().finish_loop()
        self.__loop = None
        self.__is_looping = False

    def section_enter(self, section):
        if len(self.__sections) == 0:
            raise RuntimeError("Entering a section without creating a section.\n"
                               "Always use logger.section to create a section")

        if section is not self.__sections[-1]:
            raise RuntimeError("Entering a section other than the one last_created\n"
                               "Always user with logger.section(...):")

        if len(self.__sections) > 1 and not self.__sections[-2].is_parented:
            self.__sections[-2].make_parent()
            if not self.__is_silent and not self.__sections[-1].is_silent:
                logger().log([])

        self.__log_line()

    def __log_looping_line(self):
        parts = [(f"{tracker().global_step :8,}:  ", Text.highlight)]
        parts += self.__loop.log_sections()
        parts += self.__loop_indicators
        parts += self.__loop.log_progress()

        if not self.__is_silent:
            logger().log(parts, is_new_line=False)

    def __log_line(self):
        if self.__is_looping:
            self.__log_looping_line()
            return

        if len(self.__sections) == 0:
            return

        parts = self.__sections[-1].log()
        if parts is None:
            return

        if not self.__is_silent:
            logger().log(parts, is_new_line=False)

    def set_looping_indicators(self, indicators: List[LogPart]):
        self.__loop_indicators = indicators
        self.__log_looping_line()

    def section_exit(self, section):
        if len(self.__sections) == 0:
            raise RuntimeError("Impossible")

        if section is not self.__sections[-1]:
            raise RuntimeError("Impossible")

        self.__log_line()
        self.__sections.pop(-1)


_internal: Optional[Monitor] = None


def monitor_singleton() -> Monitor:
    global _internal
    if _internal is None:
        _internal = Monitor()

    return _internal
