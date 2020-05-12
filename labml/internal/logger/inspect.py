from typing import TYPE_CHECKING, List, Tuple

from ...logger import Text

if TYPE_CHECKING:
    from . import Logger


class Inspect:
    def __init__(self, logger: 'Logger'):
        self.__logger = logger

    def _log_key_value(self, items: List[Tuple[any, any]], is_show_count=True):
        max_key_len = 0
        for k, v in items:
            max_key_len = max(max_key_len, len(str(k)))

        count = 0
        for k, v in items:
            count += 1
            spaces = " " * (max_key_len - len(str(k)))
            s = str(v)
            if len(s) > 80:
                s = f"{s[:80]} ..."
            self.__logger.log([(f"{spaces}{k}: ", Text.key),
                               (s, Text.value)])

        if is_show_count:
            self.__logger.log([
                "Total ",
                (str(count), Text.meta),
                " item(s)"])

    def info(self, *args, **kwargs):
        if len(args) == 0:
            self._log_key_value([(k, v) for k, v in kwargs.items()], False)
        elif len(args) == 1:
            assert len(kwargs.keys()) == 0
            arg = args[0]
            if type(arg) == list:
                self._log_key_value([(i, v) for i, v in enumerate(arg)])
            elif type(arg) == dict:
                self._log_key_value([(k, v) for k, v in arg.items()])
            else:
                self.__logger.log([str(arg)])
        else:
            assert len(kwargs.keys()) == 0
            self._log_key_value([(i, v) for i, v in enumerate(args)], False)
