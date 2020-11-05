from typing import List, Union, Tuple, Optional

from labml.internal.util.colors import StyleCode, ANSI_RESET
from labml.internal.logger.destinations import Destination


class ConsoleDestination(Destination):
    @staticmethod
    def __ansi_code(text: str, color: List[StyleCode] or StyleCode or None):
        """
        ### Add ansi color codes
        """
        if color is None:
            return text
        elif isinstance(color, list):
            return "".join([c.ansi() for c in color]) + f"{text}{ANSI_RESET}"
        else:
            return f"{color.ansi()}{text}{ANSI_RESET}"

    def log(self, parts: List[Union[str, Tuple[str, Optional[StyleCode]]]], *,
            is_new_line=True):
        tuple_parts = []
        for p in parts:
            if type(p) == str:
                tuple_parts.append((p, None))
            else:
                tuple_parts.append(p)
        coded = [self.__ansi_code(text, color) for text, color in tuple_parts]

        if is_new_line:
            end_char = '\n'
        else:
            end_char = ''

        text = "".join(coded)

        print("\r" + text, end=end_char, flush=True)
