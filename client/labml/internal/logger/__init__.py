from typing import Optional, List

from .destinations import Destination
from .destinations.factory import create_destination
from .inspect import Inspect
from .types import LogPart
from ..api.logs import API_LOGS as _


class Logger:
    """
    This handles the interactions among sections, loop and store
    """
    __destinations: List[Destination]

    def __init__(self):
        self.__destinations = create_destination()
        self.__inspect = Inspect(self)

    def log(self, parts: List[LogPart], *,
            is_new_line: bool = True,
            is_reset: bool = True):
        for d in self.__destinations:
            d.log(parts, is_new_line=is_new_line, is_reset=is_reset)

    def info(self, *args, **kwargs):
        self.__inspect.info(*args, **kwargs)


_internal: Optional[Logger] = None


def logger_singleton() -> Logger:
    global _internal
    if _internal is None:
        _internal = Logger()

    return _internal
