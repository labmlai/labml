from typing import List, Union, Tuple

from lab.internal.util.colors import StyleCode


class Destination:
    def log(self, parts: List[Union[str, Tuple[str, StyleCode]]], *,
            is_new_line=True):
        raise NotImplementedError()

    def new_line(self):
        raise NotImplementedError()
