import functools
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
    """
    This is similar to :func:`section` but can be used as a decorator.

    Arguments:
        name (str): Name of the function

    Keyword Arguments:
        is_silent (bool, optional): Whether not to print time taken. Defaults to ``False``.
        is_timed (bool, optional): Whether to measure time. Default to ``True``.
        is_partial (bool, optional): Whether it's a partial excution where it gets called
            repeatedly. Defaults to ``False``.
        is_new_line (bool, optional): Whether to print a new line. Defaults to ``True``.
        is_children_silent (bool, optional): Whether to make child sections silent.
            Defaults to ``True``.
        total_steps (float, optional): Total number of steps. This is used to measure progress when
            :func:`progress` gets called. Defaults to ``1``.
    """

    def decorator_func(f: Callable):
        @functools.wraps(f)
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
    """
    This creates a monitored iterator.

    Arguments:
        name (str): Name of the iterator
        iterable (Union[Iterable, Sized, int]): The iterable
        total_steps (int, optional): Total number of steps. If not provided this is
            calculated from ``iterable``.

    Keyword Arguments:
        is_silent (bool, optional): Whether not to print time taken. Defaults to ``False``.
        is_timed (bool, optional): Whether to measure time. Default to ``True``.
        is_children_silent (bool, optional): Whether to make child sections silent.
            Defaults to ``True``.
        context (optional): Reference to another section that will be used for monitoring the iteration.
    """
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
    """
    This creates a monitored enumerator.

    Arguments:
        name (str): Name of the iterator
        iterable (Sized]): The iterable

    Keyword Arguments:
        is_silent (bool, optional): Whether not to print time taken. Defaults to ``False``.
        is_timed (bool, optional): Whether to measure time. Default to ``True``.
        is_children_silent (bool, optional): Whether to make child sections silent.
            Defaults to ``True``.
        context (optional): Reference to another section that will be used for monitoring the iteration.
    """
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
    """
    This creates a monitored ``with`` block.

    Arguments:
        name (str): Name of the section

    Keyword Arguments:
        is_silent (bool, optional): Whether not to print time taken. Defaults to ``False``.
        is_timed (bool, optional): Whether to measure time. Default to ``True``.
        is_partial (bool, optional): Whether it's a partial excution where it gets called
            repeatedly. Defaults to ``False``.
        is_new_line (bool, optional): Whether to print a new line. Defaults to ``True``.
        is_children_silent (bool, optional): Whether to make child sections silent.
            Defaults to ``True``.
        total_steps (float, optional): Total number of steps. This is used to measure progress when
            :func:`progress` gets called. Defaults to ``1``.
    """
    return _internal().section(name, is_silent=is_silent,
                               is_timed=is_timed,
                               is_partial=is_partial,
                               total_steps=total_steps,
                               is_new_line=is_new_line,
                               is_children_silent=is_children_silent)


def progress(steps: float):
    """
    Set the progress of the section.

    Arguments:
        steps (float): Current progress
    """
    _internal().progress(steps)


def fail():
    """
    Mark the current section as failed.
    """
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

    This creates a monitored loop. This is designed for training loops.
    It has better monitoring than using :func:`iterate` or :func:`enum`.

    Arguments:
        iterator_ (Union[Collection, range, int]): The iterator

    Keyword Arguments:
        is_track (bool, optional): Whether track the loop time using :mod:`labml.tracker`.
            Default to ``True``.
        is_print_iteration_time (bool, optional): Whether to print iteration time. Default to ``True``.
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
    Mix a set of iterators.

    This will iterate through a list of iterators while mixing among them.
    This is useful when you want to mix training and validation steps within an epoch.
    It gives a tuple of iterator name and the element as you iterate.

    Arguments:
        total_iterations (Union[Collection, range, int]): The number of times to mix
        iterators (Tuple[str, Sized]): Are iterators and their names
        is_monit (bool, optional): Whether to monitor the iteration
    """
    return _internal().mix(total_iterations, list(iterators), is_monit=is_monit)


def finish_loop() -> None:
    """
    Finish the loop and flush all the loop related montoring stats.
    """
    _internal().finish_loop()
