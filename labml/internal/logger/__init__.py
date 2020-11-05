from typing import Optional, List

from .destinations.factory import create_destination
from .inspect import Inspect
from .types import LogPart


class Logger:
    """
    This handles the interactions among sections, loop and store
    """

    def __init__(self):
        self.__destination = create_destination()
        self.__inspect = Inspect(self)

    def log(self, parts: List[LogPart], *,
            is_new_line=True):
        self.__destination.log(parts, is_new_line=is_new_line)

    def info(self, *args, **kwargs):
        self.__inspect.info(*args, **kwargs)


_internal: Optional[Logger] = None


def logger_singleton() -> Logger:
    global _internal
    if _internal is None:
        _internal = Logger()

    return _internal
