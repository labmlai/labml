"""
Console colors
"""
from enum import Enum


class ANSICode(Enum):
    def __str__(self):
        return f"\33[{self.value}m"


class Style(ANSICode):
    normal = 0
    bold = 1
    light = 2

    italic = 3  # Not supported in PyCharm
    underline = 4

    highlight = 7


class Color(ANSICode):
    black = 30
    red = 31
    green = 32
    orange = 33
    blue = 34
    purple = 35
    cyan = 36
    white = 37


class BrightColor(ANSICode):
    black = 90
    red = 91
    green = 92
    orange = 93
    blue = 94
    purple = 95
    cyan = 96
    white = 97

    gray = 37


class Background(ANSICode):
    black = 40
    red = 41
    green = 42
    orange = 43
    blue = 44
    purple = 45
    cyan = 46
    white = 47


class BrightBackground(ANSICode):
    black = 100
    red = 101
    green = 102
    orange = 103
    blue = 104
    purple = 105
    cyan = 106
    white = 107


Reset = "\33[0m"

if __name__ == "__main__":
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
        for c in Color:
            for b in Background:
                print(
                    f"{s}{c}{b}{s.name}, {c.name}, {b.name}{Reset}")

    print(Style.bold.name)
