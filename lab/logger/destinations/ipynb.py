from typing import List, Union, Tuple

from lab.logger.colors import StyleCode
from lab.logger.destinations import Destination


class IpynbDestination(Destination):
    @staticmethod
    def __html_code(text: str, color: List[StyleCode] or StyleCode or None):
        """
        ### Add ansi color codes
        """
        if color is None:
            return text
        elif type(color) is list:
            style = "".join(color.html_style())
        else:
            style = color.html_style()

        return f"<span style=\"{style}\">{text}</span>"

    def log(self, parts: List[Union[str, Tuple[str, StyleCode]]], *,
            is_new_line=True):
        tuple_parts = []
        for p in parts:
            if type(p) == str:
                tuple_parts.append((p, None))
            else:
                tuple_parts.append(p)
        coded = [self.__html_code(text, color) for text, color in tuple_parts]

        if is_new_line:
            end_char = '\n'
        else:
            end_char = ''

        text = "".join(coded)

        print("\r" + text, end=end_char, flush=True)

    def new_line(self):
        print()
