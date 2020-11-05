from typing import List, Union, Tuple, Optional

from labml.internal.util.colors import StyleCode


class Destination:
    def log(self, parts: List[Union[str, Tuple[str, Optional[StyleCode]]]], *,
            is_new_line=True):
        raise NotImplementedError()
