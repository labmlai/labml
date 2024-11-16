from typing import List, Union, Tuple, Optional

from ...app.logs import APP_CONSOLE_LOGS
from ..destinations import Destination
from ...util.colors import StyleCode, ANSI_RESET


class ConsoleDestination(Destination):
    def __init__(self, is_screen: bool):
        self.is_screen = is_screen

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
            is_new_line: bool,
            is_reset: bool):
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

        if is_reset:
            if text:
                self.print("\r" + " " * 100 + "\r" + text, end_char)
            else:
                self.print("\r" + text, end_char)
        else:
            self.print(text, end_char)

    def print(self, text: str, end_char: str):
        text = text.encode('utf-8', 'xmlcharrefreplace').decode('utf-8')
        if self.is_screen:
            print(text, end=end_char, flush=True)
        else:
            APP_CONSOLE_LOGS.outputs(logger_=text + end_char)
