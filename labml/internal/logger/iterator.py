import typing
from typing import Optional, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from labml.internal.logger import Logger
from labml.internal.logger.sections import Section


class Iterator:
    r"""
        Note:
            You should use :meth:`labml.logger.iterate`
            or :meth:`labml.monit.enum` to create iterators
            and enumerators.
    """

    _section: Optional[Section]

    def __init__(self, *,
                 logger: 'Logger',
                 name: str,
                 iterable: typing.Union[Iterable, typing.Sized, int],
                 is_silent: bool,
                 is_timed: bool,
                 total_steps: Optional[int],
                 is_enumerate: bool):
        if is_enumerate:
            total_steps = len(iterable)
            iterable = enumerate(iterable)
        if type(iterable) is int:
            total_steps = iterable
            iterable = range(total_steps)
        if total_steps is None:
            sized: typing.Sized = iterable
            total_steps = len(sized)

        self._logger = logger
        self._name = name
        self._iterable: Iterable = iterable
        self._iterator = Optional[typing.Iterator]
        self._total_steps = total_steps
        self._section = None
        self._is_silent = is_silent
        self._is_timed = is_timed
        self._counter = -1

    def get_estimated_time(self) -> float:
        if self._section:
            return self._section.get_estimated_time()
        else:
            return 0

    def __iter__(self):
        self._section = self._logger.section(
            self._name,
            is_silent=self._is_silent,
            is_timed=self._is_timed,
            is_partial=False,
            total_steps=self._total_steps,
            is_new_line=True)
        self._iterator = iter(self._iterable)
        self._section.__enter__()

        return self

    def __next__(self):
        try:
            self._counter += 1
            self._logger.progress(self._counter)
            next_value = next(self._iterator)
        except StopIteration as e:
            self._section.__exit__(None, None, None)
            raise e

        return next_value
