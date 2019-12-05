"""
# Logger class

*Should be initialized via `Experiment`*

This module contains logging and monitoring helpers.

Logger prints to the screen and writes TensorBoard summaries.
"""
import typing
from typing import List, Tuple, Optional

from lab.colors import ANSICode
from .internal import LoggerInternal as _LoggerInternal
from .iterator import Iterator
from .delayed_keyboard_interrupt import DelayedKeyboardInterrupt
from .indicators import IndicatorType, IndicatorOptions, Indicator
from .loop import Loop
from .sections import Section, OuterSection, LoopingSection, section_factory
from .store import Store
from .writers import Writer, ScreenWriter

logger_singleton: 'Logger' = None


class Logger:
    """
    ## 🖨 Logger class
    """

    def __init__(self):
        """
        ### Initializer
        """
        if logger_singleton is not None:
            raise RuntimeError("Only one instance of logger can exist")

        self.internal = _LoggerInternal()

    def log(self, message, *,
            color: List[ANSICode] or ANSICode or None = None,
            new_line=True):
        """
        ### Print a message to screen in color
        """

        self.internal.log(message, color=color, new_line=new_line)

    def log_color(self, parts: List[Tuple[str, ANSICode or None]], *,
                  new_line=True):
        """
        ### Print a message with different colors.
        """

        self.internal.log_color(parts, new_line=new_line)

    def add_indicator(self, name: str,
                      type_: IndicatorType,
                      options: IndicatorOptions = None):
        """
        ### Add an indicator
        """

        self.internal.add_indicator(name, type_, options)

    def store(self, *args, **kwargs):
        """
        ### Stores a value in the logger_class.

        This may be added to a queue, a list or stored as
        a TensorBoard histogram depending on the
        type of the indicator.
        """

        self.internal.store(*args, **kwargs)

    def set_global_step(self, global_step):
        self.internal.set_global_step(global_step)

    def add_global_step(self, global_step: int = 1):
        self.internal.add_global_step(global_step)

    def new_line(self):
        self.internal.new_line()

    def write(self):
        """
        ### Output the stored log values to screen and TensorBoard summaries.
        """

        self.internal.write()

    def save_checkpoint(self, *args):
        self.internal.save_checkpoint(*args)

    def iterator(self, name, iterable: typing.Union[typing.Iterable, typing.Sized, int],
                 total_steps: Optional[int] = None, *,
                 is_silent: bool = False,
                 is_timed: bool = True):
        return self.internal.iterator(name, iterable, total_steps, is_silent=is_silent,
                                      is_timed=is_timed)

    def enumerator(self, name, iterable: typing.Sized, *,
                   is_silent: bool = False,
                   is_timed: bool = True):
        return self.internal.enumerator(name, iterable, is_silent=is_silent, is_timed=is_timed)

    def section(self, name, *,
                is_silent: bool = False,
                is_timed: bool = True,
                is_partial: bool = False,
                total_steps: float = 1.0):
        return self.internal.section(name, is_silent=is_silent,
                                     is_timed=is_timed,
                                     is_partial=is_partial,
                                     total_steps=total_steps)

    def progress(self, steps: float):
        self.internal.progress(steps)

    def set_successful(self, is_successful=True):
        self.internal.set_successful(is_successful)

    def loop(self, iterator_: range, *,
             is_print_iteration_time=True):
        return self.internal.loop(iterator_, is_print_iteration_time=is_print_iteration_time)

    def finish_loop(self):
        self.internal.finish_loop()

    def delayed_keyboard_interrupt(self):
        """
        ### Create a section with a delayed keyboard interrupt
        """
        return self.internal.delayed_keyboard_interrupt()

    def info(self, *args, **kwargs):
        """
        ### 🎨 Pretty prints a set of values.
        """

        self.internal.info(*args, **kwargs)


logger_singleton: 'Logger' = Logger()
