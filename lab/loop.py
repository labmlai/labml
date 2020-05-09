from typing import Union, Optional, overload

from lab.internal.logger import logger_singleton as _internal


def set_global_step(global_step: Optional[int]):
    _internal().set_global_step(global_step)


def add_global_step(increment_global_step: int = 1):
    _internal().add_global_step(int(increment_global_step))


def get_global_step() -> int:
    return _internal().global_step


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
