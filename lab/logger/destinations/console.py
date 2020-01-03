from typing import List, Union, Tuple

from lab.logger.colors import ANSICode, Reset
from lab.logger.destinations import Destination


class ConsoleDestination(Destination):
    @staticmethod
    def __ansi_code(text: str, color: List[ANSICode] or ANSICode or None):
        """
        ### Add ansi color codes
        """
        if color is None:
            return text
        elif type(color) is list:
            return "".join(color) + f"{text}{Reset}"
        else:
            return f"{color}{text}{Reset}"

    def log(self, parts: List[Union[str, Tuple[str, ANSICode]]], *,
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

    def new_line(self):
        print()
