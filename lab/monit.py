from typing import Union, Optional, Iterable, Sized

from lab.internal.logger import logger_singleton as _internal


def iterate(name, iterable: Union[Iterable, Sized, int],
            total_steps: Optional[int] = None, *,
            is_silent: bool = False,
            is_timed: bool = True):
    return _internal().iterate(name, iterable, total_steps,
                               is_silent=is_silent,
                               is_timed=is_timed)


def enum(name, iterable: Sized, *,
         is_silent: bool = False,
         is_timed: bool = True):
    return _internal().enum(name, iterable, is_silent=is_silent, is_timed=is_timed)


def section(name, *,
            is_silent: bool = False,
            is_timed: bool = True,
            is_partial: bool = False,
            is_new_line: bool = True,
            total_steps: float = 1.0):
    return _internal().section(name, is_silent=is_silent,
                               is_timed=is_timed,
                               is_partial=is_partial,
                               total_steps=total_steps,
                               is_new_line=is_new_line)


def progress(steps: float):
    _internal().progress(steps)


def fail():
    _internal().set_successful(False)
