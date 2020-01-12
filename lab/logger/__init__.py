from typing import Union, List, Tuple, Optional, Iterable, Sized

import numpy as np

from .colors import StyleCode
from .indicators import Indicator
from .internal import LoggerInternal as _LoggerInternal

_internal: Optional[_LoggerInternal] = None


def internal() -> _LoggerInternal:
    global _internal
    if _internal is None:
        _internal = _LoggerInternal()

    return _internal


def log(message: Union[str, List[Union[str, Tuple[str, StyleCode]]]],
        color: List[StyleCode] or StyleCode or None = None,
        *,
        is_new_line=True):
    if type(message) == str:
        internal().log([(message, color)], is_new_line=is_new_line)
    elif type(message) == list:
        internal().log(message, is_new_line=is_new_line)


def add_indicator(indicator: Indicator):
    internal().add_indicator(indicator)


def store(*args, **kwargs):
    """
    ### Stores a value in the logger.

    This may be added to a queue, a list or stored as
    a TensorBoard histogram depending on the
    type of the indicator.
    """

    internal().store(*args, **kwargs)


def write():
    """
    ### Output the stored log values to screen and TensorBoard summaries.
    """

    internal().write()


def new_line():
    internal().new_line()


def set_global_step(global_step):
    internal().set_global_step(global_step)


def add_global_step(global_step: int = 1):
    internal().add_global_step(global_step)


def get_global_step() -> int:
    return internal().global_step


def iterate(name, iterable: Union[Iterable, Sized, int],
            total_steps: Optional[int] = None, *,
            is_silent: bool = False,
            is_timed: bool = True):
    return internal().iterate(name, iterable, total_steps, is_silent=is_silent,
                              is_timed=is_timed)


def enum(name, iterable: Sized, *,
         is_silent: bool = False,
         is_timed: bool = True):
    return internal().enum(name, iterable, is_silent=is_silent, is_timed=is_timed)


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


def save_checkpoint():
    internal().save_checkpoint()


def info(*args, **kwargs):
    """
    ### 🎨 Pretty prints a set of values.
    """

    internal().info(*args, **kwargs)


def get_data_path():
    return internal().get_data_path()


def save_numpy(name: str, array: np.ndarray):
    """
    ## Save a single numpy array

    This is used to save processed data
    """
    internal().save_numpy(name, array)
