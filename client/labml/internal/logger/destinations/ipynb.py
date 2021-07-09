import threading
from typing import List, Union, Tuple, Callable, Optional

from IPython.core.display import display, HTML

from labml.internal.logger.destinations import Destination
from labml.internal.util import is_kaggle
from labml.internal.util.colors import StyleCode

get_ipython: Callable


class IpynbDestination(Destination):
    def __init__(self):
        self.__last_handle = None
        self.__last_id = 1
        self.__cell_lines = []
        self.__cell_count = 0
        self.lock = threading.Lock()

    def is_same_cell(self):
        try:
            cells = get_ipython().ev('len(In)')
        except NameError:
            return False

        if cells == self.__cell_count:
            return True

        self.__cell_count = cells
        self.__cell_lines = []
        self.__last_handle = None

        return False

    @staticmethod
    def html_code(text: str, color: List[StyleCode] or StyleCode or None):
        """
        ### Add ansi color codes
        """
        if text == '\n':
            assert color is None
            return text

        if color is None:
            return text
        elif isinstance(color, list):
            open_tags = ''.join([c.html_open(text) for c in color])
            close_tags = ''.join([c.html_close(text) for c in reversed(color)])
        else:
            open_tags = color.html_open(text)
            close_tags = color.html_close(text)

        return open_tags + text + close_tags

    def log(self, parts: List[Union[str, Tuple[str, Optional[StyleCode]]]], *,
            is_new_line: bool,
            is_reset: bool):
        tuple_parts = []
        for p in parts:
            if type(p) == str:
                text = p
                style = None
            else:
                text, style = p
            lines = text.split('\n')
            for line in lines[:-1]:
                tuple_parts.append((line, style))
                tuple_parts.append(('\n', None))
            tuple_parts.append((lines[-1], style))

        coded = [self.html_code(text, color) for text, color in tuple_parts]

        text = "".join(coded)
        lines = text.split('\n')

        if is_kaggle():
            attrs = 'style="color: #444; overflow-x: scroll;"'
        else:
            attrs = 'style="overflow-x: scroll;"'

        with self.lock:
            if self.is_same_cell():
                if coded:
                    last = self.__cell_lines.pop()
                    if is_reset:
                        self.__cell_lines += lines
                    else:
                        self.__cell_lines += [last + lines[0]] + lines[1:]
                text = '\n'.join(self.__cell_lines)
                html = HTML(f"<pre {attrs}>{text}</pre>")
                self.__last_handle.update(html)
            else:
                self.__cell_lines = lines
                text = '\n'.join(self.__cell_lines)
                html = HTML(f"<pre {attrs}>{text}</pre>")
                self.__last_handle = display(html, display_id=self.__last_id)
                self.__last_id += 1

            # print(len(self.__cell_lines), self.__cell_lines[-1], is_new_line)
            if is_new_line:
                self.__cell_lines.append('')
