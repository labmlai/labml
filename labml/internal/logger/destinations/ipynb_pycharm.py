import threading
from typing import List, Union, Tuple, Callable, Optional

from IPython.core.display import display
from ipywidgets import HTML

from labml.internal.util.colors import StyleCode
from labml.internal.logger.destinations import Destination
from labml.internal.logger.destinations.ipynb import IpynbDestination
from labml.internal.util import is_kaggle

get_ipython: Callable


class IpynbPyCharmDestination(Destination):
    def __init__(self):
        self.__cell_lines = []
        self.__cell_count = 0
        self.html: Optional[HTML] = None
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

        return False

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

        coded = [IpynbDestination.html_code(text, color) for text, color in tuple_parts]

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
                self.html.value = f"<pre {attrs}>{text}</pre>"
            else:
                self.__cell_lines = lines
                text = '\n'.join(self.__cell_lines)
                self.html = HTML(f"<pre  {attrs}>{text}</pre>")
                display(self.html)

            # print(len(self.__cell_lines), self.__cell_lines[-1], is_new_line)
            if is_new_line:
                self.__cell_lines.append('')
