import sys
from enum import Enum

from labml import logger
from labml.logger import Text

_CHARS_PER_DOT = 128


class UIMode(Enum):
    none = 'none'
    dots = 'dots'
    full = 'full'

    def on_bytes(self, data: bytes, *, is_err: bool):
        if self == UIMode.full:
            if is_err:
                sys.stdout.write(data.decode('utf-8'))
            else:
                sys.stderr.write(data.decode('utf-8'))
        elif self == UIMode.dots:
            count = len(data) / _CHARS_PER_DOT
            if 0 < count < 1:
                count = 1
            count = round(count)
            if count == 0:
                return
            if is_err:
                logger.log('.' * count, Text.warning, is_new_line=False, is_reset=False)
            else:
                logger.log('.' * count, Text.meta, is_new_line=False, is_reset=False)

    def end(self):
        if self != UIMode.none:
            logger.log()
