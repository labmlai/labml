"""
# Logger class

*Should be initialized via `Experiment`*

This module contains logging and monitoring helpers.

Logger prints to the screen and writes TensorBoard summaries.
"""

import typing

from .colors import ANSICode
from .indicators import IndicatorType, IndicatorOptions
from .internal import LoggerInternal as _LoggerInternal

_internal: typing.Optional[_LoggerInternal] = None


def internal() -> _LoggerInternal:
    global _internal
    if _internal is None:
        _internal = _LoggerInternal()

    return _internal


def log(message, *,
        color: typing.List[ANSICode] or ANSICode or None = None,
        new_line=True):
    """
    ### Print a message to screen in color
    """

    internal().log(message, color=color, new_line=new_line)


def log_color(parts: typing.List[typing.Union[str, typing.Tuple[str, ANSICode]]], *,
              new_line=True):
    """
    ### Print a message with different colors.
    """

    internal().log_color(parts, new_line=new_line)


def add_indicator(name: str,
                  type_: IndicatorType,
                  options: IndicatorOptions = None):
    """
    ### Add an indicator
    """

    internal().add_indicator(name, type_, options)


def store(*args, **kwargs):
    """
    ### Stores a value in the logger.

    This may be added to a queue, a list or stored as
    a TensorBoard histogram depending on the
    type of the indicator.
    """

    internal().store(*args, **kwargs)


def set_global_step(global_step):
    internal().set_global_step(global_step)


def add_global_step(global_step: int = 1):
    internal().add_global_step(global_step)


def new_line():
    internal().new_line()


def write():
    """
    ### Output the stored log values to screen and TensorBoard summaries.
    """

    internal().write()


def save_checkpoint():
    internal().save_checkpoint()


def iterator(name, iterable: typing.Union[typing.Iterable, typing.Sized, int],
             total_steps: typing.Optional[int] = None, *,
             is_silent: bool = False,
             is_timed: bool = True):
    return internal().iterator(name, iterable, total_steps, is_silent=is_silent,
                               is_timed=is_timed)


def enumerator(name, iterable: typing.Sized, *,
               is_silent: bool = False,
               is_timed: bool = True):
    return internal().enumerator(name, iterable, is_silent=is_silent, is_timed=is_timed)


def section(name, *,
            is_silent: bool = False,
            is_timed: bool = True,
            is_partial: bool = False,
            total_steps: float = 1.0):
    return internal().section(name, is_silent=is_silent,
                              is_timed=is_timed,
                              is_partial=is_partial,
                              total_steps=total_steps)


def progress(steps: float):
    internal().progress(steps)


def set_successful(is_successful=True):
    internal().set_successful(is_successful)


def loop(iterator_: range, *,
         is_print_iteration_time=True):
    return internal().loop(iterator_, is_print_iteration_time=is_print_iteration_time)


def finish_loop():
    internal().finish_loop()


def delayed_keyboard_interrupt():
    """
    ### Create a section with a delayed keyboard interrupt
    """
    return internal().delayed_keyboard_interrupt()


def info(*args, **kwargs):
    """
    ### 🎨 Pretty prints a set of values.
    """

    internal().info(*args, **kwargs)
