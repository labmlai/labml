from typing import List, Union, Tuple

from lab.logger.colors import ANSICode


class Destination:
    def log(self, parts: List[Union[str, Tuple[str, ANSICode]]], *,
            is_new_line=True):
        raise NotImplementedError()

    def new_line(self):
        raise NotImplementedError()
