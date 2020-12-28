from typing import Iterable, Sized, Collection, Callable, Tuple
from typing import Union, Optional, overload

from labml.internal.monitor import monitor_singleton as _internal


def clear():
    _internal().clear()


def func(name, *,
         is_silent: bool = False,
         is_timed: bool = True,
         is_partial: bool = False,
         is_new_line: bool = True,
         is_children_silent: bool = False,
         total_steps: float = 1.0):
    def decorator_func(f: Callable):
        def wrapper(*args, **kwargs):
            with section(name,
                         is_silent=is_silent,
                         is_timed=is_timed,
                         is_partial=is_partial,
                         is_new_line=is_new_line,
                         is_children_silent=is_children_silent,
                         total_steps=total_steps):
                return f(*args, **kwargs)

        return wrapper

    return decorator_func


def iterate(name, iterable: Union[Iterable, Sized, int],
            total_steps: Optional[int] = None, *,
            is_silent: bool = False,
            is_children_silent: bool = False,
            is_timed: bool = True,
            context=None):
    return _internal().iterate(name, iterable, total_steps,
                               is_silent=is_silent,
                               is_children_silent=is_children_silent,
                               is_timed=is_timed,
                               section=context)


def enum(name, iterable: Sized, *,
         is_silent: bool = False,
         is_children_silent: bool = False,
         is_timed: bool = True,
         context=None):
    return _internal().enum(name, iterable,
                            is_silent=is_silent,
                            is_children_silent=is_children_silent,
                            is_timed=is_timed,
                            section=context)


def section(name, *,
            is_silent: bool = False,
            is_timed: bool = True,
            is_partial: bool = False,
            is_new_line: bool = True,
            is_children_silent: bool = False,
            total_steps: float = 1.0):
    return _internal().section(name, is_silent=is_silent,
                               is_timed=is_timed,
                               is_partial=is_partial,
                               total_steps=total_steps,
                               is_new_line=is_new_line,
                               is_children_silent=is_children_silent)


def progress(steps: float):
    _internal().progress(steps)


def fail():
    _internal().set_successful(False)


@overload
def loop(iterator_: int, *,
         is_track: bool = True,
         is_print_iteration_time: bool = True):
    ...


@overload
def loop(iterator_: range, *,
         is_track: bool = True,
         is_print_iteration_time: bool = True):
    ...


@overload
def loop(iterator_: Collection, *,
         is_track: bool = True,
         is_print_iteration_time: bool = True):
    ...


def loop(iterator_: Union[Collection, range, int], *,
         is_track: bool = True,
         is_print_iteration_time: bool = True):
    """
        This has multiple overloads

        .. function:: loop(iterator_: range, *, is_track=True, is_print_iteration_time=True)
            :noindex:

        .. function:: loop(iterator_: int, *, is_track=True, is_print_iteration_time=True)
            :noindex:
        """

    if type(iterator_) == int:
        return _internal().loop(range(iterator_),
                                is_track=is_track,
                                is_print_iteration_time=is_print_iteration_time)
    else:
        return _internal().loop(iterator_,
                                is_track=is_track,
                                is_print_iteration_time=is_print_iteration_time)


def mix(total_iterations, *iterators: Tuple[str, Sized],
        is_monit: bool = True):
    """
    Mix a set of iterators
    """
    return _internal().mix(total_iterations, list(iterators), is_monit=is_monit)


def finish_loop():
    _internal().finish_loop()
