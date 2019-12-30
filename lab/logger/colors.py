"""
Console colors
"""
import warnings
from enum import Enum


def __getattr__(name):
    if name == 'BrightColor':
        warnings.warn("BrightColor is unsupported in iTerm2, dull on Jupyter"
                      " and looks bad on white background")
        return _BrightColor
    elif name == 'BrightBackground':
        warnings.warn("BrightBackground is unsupported in iTerm2, dull on Jupyter"
                      " and looks bad on white background")
        return _BrightBackground


class ANSICode(Enum):
    def __str__(self):
        if self.value is None:
            return ""
        elif type(self.value) == int:
            return f"\33[{self.value}m"
        elif type(self.value) == list:
            return ''.join([f"\33[{v}m" for v in self.value])
        else:
            assert False


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


class Text(ANSICode):
    none = None
    danger = Color.red.value
    success = Color.green.value
    warning = Color.orange.value
    meta = Color.blue.value
    key = Color.cyan.value
    meta2 = Color.purple.value
    title = [Style.bold.value, Style.underline.value]
    heading = Style.underline.value
    value = Style.bold.value
    highlight = [Style.bold.value, Color.orange.value]
    subtle = [Style.light.value, Color.white.value]


class _BrightColor(ANSICode):
    none = None

    black = 90
    red = 91
    green = 92
    orange = 93
    blue = 94
    purple = 95
    cyan = 96
    white = 97


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


class _BrightBackground(ANSICode):  # - iTerm2 - (doesn't work well with white bg)
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
