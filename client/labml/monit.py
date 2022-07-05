import functools
from typing import Iterable, Sized, Collection, Callable, Tuple, Any
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
         is_track: bool = False,
         is_not_in_loop: bool = False,
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
        is_track (bool, optional): Whether to track the time.
            Defaults to ``False``.
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
                         is_track=is_track,
                         is_not_in_loop=is_not_in_loop,
                         total_steps=total_steps):
                return f(*args, **kwargs)

        return wrapper

    return decorator_func


def iterate(name, iterable: Union[Iterable, Sized, int],
            total_steps: Optional[int] = None, *,
            is_silent: bool = False,
            is_children_silent: bool = False,
            is_timed: bool = True,
            is_track: bool = False,
            is_not_in_loop: bool = False,
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
        is_track (bool, optional): Whether to track the time.
            Defaults to ``False``.
        context (optional): Reference to another section that will be used for monitoring the iteration.
    """
    return _internal().iterate(name, iterable, total_steps,
                               is_silent=is_silent,
                               is_children_silent=is_children_silent,
                               is_timed=is_timed,
                               is_track=is_track,
                               is_not_in_loop=is_not_in_loop,
                               section=context)


def enum(name, iterable: Sized, *,
         is_silent: bool = False,
         is_children_silent: bool = False,
         is_timed: bool = True,
         is_track: bool = False,
         is_not_in_loop: bool = False,
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
        is_track (bool, optional): Whether to track the time.
            Defaults to ``False``.
        context (optional): Reference to another section that will be used for monitoring the iteration.
    """
    return _internal().enum(name, iterable,
                            is_silent=is_silent,
                            is_children_silent=is_children_silent,
                            is_timed=is_timed,
                            is_track=is_track,
                            is_not_in_loop=is_not_in_loop,
                            section=context)


def section(name, *,
            is_silent: bool = False,
            is_timed: bool = True,
            is_partial: bool = False,
            is_new_line: bool = True,
            is_children_silent: bool = False,
            is_track: bool = False,
            is_not_in_loop: bool = False,
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
        is_track (bool, optional): Whether to track the time.
            Defaults to ``False``.
        total_steps (float, optional): Total number of steps. This is used to measure progress when
            :func:`progress` gets called. Defaults to ``1``.
    """
    return _internal().section(name, is_silent=is_silent,
                               is_timed=is_timed,
                               is_partial=is_partial,
                               total_steps=total_steps,
                               is_new_line=is_new_line,
                               is_track=is_track,
                               is_not_in_loop=is_not_in_loop,
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


@overload
def mix(*iterators: Tuple[Union[str, Callable[[Any], None]], Union[Sized, int]],
        is_monit: bool = True):
    ...


@overload
def mix(total_iterations: int, *iterators: Tuple[Union[str, Callable[[Any], None]], Union[Sized, int]],
        is_monit: bool = True):
    ...


def mix(*args,
        is_monit: bool = True):
    """
    This has two overloads

    .. function:: mix(*iterators: Tuple[Union[str, Callable[[Any], None]], Union[Sized, int]], is_monit: bool = True)
        :noindex:

    .. function:: mix(total_iterations: int, *iterators: Tuple[Union[str, Callable[[Any], None]], Union[Sized, int]], is_monit: bool = True):
        :noindex:

    This will iterate through a list of iterators while mixing among them.
    This is useful, for instance, when you want to mix training, validation, sampling steps within an epoch.

    You can give it tuples of iterator names and iterators.
    It will yield the names along with the iterator values.

    If you pass a function instead of a name it will call that function with the iterator value, instead of
    yielding it.

    Arguments:
        total_iterations (Union[Collection, range, int]): The number of times to mix
        iterators (Tuple[Union[str, Callable[[Any], None]], Union[Sized, int]]): Are iterators and their names or callback function
        is_monit (bool, optional): Whether to monitor the iterations, when inside :func:`loop`. Default to ``True``.
    """

    total_iterations = None
    iterators = []
    for arg in args:
        if isinstance(arg, tuple):
            r, it = arg
            if isinstance(it, int):
                it = range(it)
            iterators.append((r, it))
        elif isinstance(arg, int):
            total_iterations = arg
        else:
            raise ValueError(f'Unknown argument type: {type(arg)}, {arg}')

    if total_iterations is None:
        total_iterations = max(len(iterator[1]) for iterator in iterators)

    return _internal().mix(total_iterations, list(iterators), is_monit=is_monit)


def finish_loop() -> None:
    """
    Finish the loop and flush all the loop related monitoring stats.
    """
    _internal().finish_loop()
