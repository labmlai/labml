from typing import List, Union, Tuple

from IPython.core.display import display, HTML

from lab.logger.colors import StyleCode
from lab.logger.destinations import Destination


class IpynbDestination(Destination):
    def __init__(self):
        self.__last_handle = None
        self.__last_id = 0

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

        text = "".join(coded)

        html = HTML(f"<pre>{text}</pre>")

        if self.__last_handle is not None:
            self.__last_handle.update(html)
        else:
            self.__last_handle = display(html, display_id=self.__last_id)
            self.__last_id += 1

        if is_new_line:
            self.__last_handle = None

    def new_line(self):
        self.__last_handle = None
