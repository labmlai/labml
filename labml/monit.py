from typing import Iterable, Sized
from typing import Union, Optional, overload

from labml.internal.logger import logger_singleton as _internal


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


@overload
def loop(iterator_: int, *,
         is_print_iteration_time: bool = True):
    ...


@overload
def loop(iterator_: range, *,
         is_print_iteration_time: bool = True):
    ...


def loop(iterator_: Union[range, int], *,
         is_print_iteration_time: bool = True):
    """
        This has multiple overloads

        .. function:: loop(iterator_: range, *,is_print_iteration_time=True)
            :noindex:

        .. function:: loop(iterator_: int, *,is_print_iteration_time=True)
            :noindex:
        """

    if type(iterator_) == int:
        return _internal().loop(range(iterator_), is_print_iteration_time=is_print_iteration_time)
    else:
        return _internal().loop(iterator_, is_print_iteration_time=is_print_iteration_time)


def finish_loop():
    _internal().finish_loop()
