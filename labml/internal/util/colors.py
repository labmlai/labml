"""
Console colors
"""
from enum import Enum

_ANSI_CODES = dict(
    normal=0,
    bold=1,
    light=2,  # - PyCharm/Jupyter

    italic=3,  # - PyCharm/Jupyter
    underline=4,

    highlight=7,  # Changes background in PyCharm/Terminal

    # Colors
    black=30,
    red=31,
    green=32,
    orange=33,
    blue=34,
    purple=35,
    cyan=36,
    white=37,

    # Background [Not used anymore]
    bg_black=40,
    bg_red=41,
    bg_green=42,
    bg_orange=43,
    bg_blue=44,
    bg_purple=45,
    bg_cyan=46,
    bg_white=47,

    # Bright Colors [Not used anymore]
    bright_black=90,
    bright_red=91,
    bright_green=92,
    bright_orange=93,
    bright_blue=94,
    bright_purple=95,
    bright_cyan=96,
    bright_white=97,

    # Bright Background Colors [Not used anymore]
    bg_bright_black=100,
    bg_bright_red=101,
    bg_bright_green=102,
    bg_bright_orange=103,
    bg_bright_blue=104,
    bg_bright_purple=105,
    bg_bright_cyan=106,
    bg_bright_white=107
)

ANSI_RESET = "\33[0m"

_HTML_STYLES = dict(
    normal=('', ''),
    bold=('<strong>', '</strong>'),
    underline=('<span style="text-decoration: underline">', '</span>'),
    light=('', ''),

    # Colors
    black=('<span style="color: #3E424D">', '</span>'),
    red=('<span style="color: #E75C58">', '</span>'),
    green=('<span style="color: #00A250">', '</span>'),
    orange=('<span style="color: #DDB62B">', '</span>'),
    blue=('<span style="color: #208FFB">', '</span>'),
    purple=('<span style="color: #D160C4">', '</span>'),
    cyan=('<span style="color: #60C6C8">', '</span>'),
    white=('<span style="color: #C5C1B4">', '</span>')
)


class StyleCode(Enum):
    r"""
    This is the base class for different style enumerations
    """

    def ansi(self):
        if self.value is None:
            return f"\33[{_ANSI_CODES['normal']}m"
        elif type(self.value) == str:
            return f"\33[{_ANSI_CODES[self.value]}m"
        elif type(self.value) == list:
            return ''.join([f"\33[{_ANSI_CODES[v]}m" for v in self.value])
        else:
            assert False

    def html_open(self):
        if self.value is None:
            return ""
        elif type(self.value) == str:
            return _HTML_STYLES[self.value][0]
        elif type(self.value) == list:
            return ''.join([_HTML_STYLES[v][0] for v in self.value])
        else:
            assert False

    def html_close(self):
        if self.value is None:
            return ""
        elif type(self.value) == str:
            return _HTML_STYLES[self.value][1]
        elif type(self.value) == list:
            return ''.join([_HTML_STYLES[v][1] for v in reversed(self.value)])
        else:
            assert False


def _test():
    for i in [0, 38, 48]:
        for j in [5]:
            for k in range(16):
                print("\33[{};{};{}m{:02d},{},{:03d}\33[0m\t".format(i, j, k, i, j, k),
                      end='')
                if (k + 1) % 6 == 0:
                    print("")
            print("")

    for i in range(0, 128):
        print(f"\33[{i}m{i :03d}\33[0m ", end='')
        if (i + 1) % 10 == 0:
            print("")

    print()

    print("▁▂▃▄▅▆▇█")
    print("▁▂▃▄▅▆▇█")


if __name__ == "__main__":
    _test()
