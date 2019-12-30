"""
Console colors
"""
from enum import Enum


class ANSICode(Enum):
    def __str__(self):
        if self.value is None:
            return ""
        else:
            return f"\33[{self.value}m"


class Style(ANSICode):
    none = None
    normal = 0
    bold = 1
    light = 2  # - PyCharm/Jupyter

    italic = 3  # - PyCharm/Jupyter
    underline = 4

    highlight = 7  # Changes background in PyCharm/Terminal


class Color(ANSICode):
    none = None
    black = 30
    red = 31
    green = 32
    orange = 33
    blue = 34
    purple = 35
    cyan = 36
    white = 37


class BrightColor(ANSICode):
    # - iTerm2
    # - doesn't work well with white bg
    # - Dull on Jupyter
    none = None

    black = 90
    red = 91
    green = 92
    orange = 93
    blue = 94
    purple = 95
    cyan = 96
    white = 97

    gray = 37  # - Terminal/Jupyter


class Background(ANSICode):
    none = None

    black = 40
    red = 41
    green = 42
    orange = 43
    blue = 44
    purple = 45
    cyan = 46
    white = 47


class BrightBackground(ANSICode):  # - iTerm2 - (doesn't work well with white bg)
    none = None

    black = 100
    red = 101
    green = 102
    orange = 103
    blue = 104
    purple = 105
    cyan = 106
    white = 107


Reset = "\33[0m"


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

    print("")

    for s in Style:
        for i, c in enumerate(Color):
            for j, b in enumerate(Background):
                if s.name != 'none' and (i > 1 or j > 1):
                    continue
                if c.name != 'none' and (j > 0):
                    continue
                print(
                    f"{s}{c}{b}{s.name}, {c.name}, {b.name}{Reset}")

    for s in Style:
        for i, c in enumerate(BrightColor):
            for j, b in enumerate(BrightBackground):
                if s.name != 'none' and (i > 1 or j > 1):
                    continue
                if c.name != 'none' and (j > 0):
                    continue
                print(
                    f"{s}{c}{b}{s.name}, {c.name}, {b.name}{Reset}")


if __name__ == "__main__":
    _test()
