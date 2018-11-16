"""
Console colors
"""


class Style:
    bold = "[00;5;001m"
    light = "[00;5;002m"

    italic = "[00;5;003m"
    underline = "[00;5;004m"

    strike = "[00;5;009m"


class Color:
    black = "[38;5;000m"
    red = "[38;5;001m"
    green = "[38;5;002m"
    orange = "[38;5;003m"
    blue = "[38;5;004m"
    purple = "[38;5;005m"
    cyan = "[38;5;006m"
    white = "[38;5;007m"


class BrightColor:
    black = "[38;5;008m"
    red = "[38;5;009m"
    green = "[38;5;010m"
    orange = "[38;5;011m"
    blue = "[38;5;012m"
    purple = "[38;5;013m"
    cyan = "[38;5;014m"
    white = "[38;5;015m"


class Background:
    black = "[48;5;000m"
    red = "[48;5;001m"
    green = "[48;5;002m"
    orange = "[48;5;003m"
    blue = "[48;5;004m"
    purple = "[48;5;005m"
    cyan = "[48;5;006m"
    white = "[48;5;007m"


class BrightBackground:
    black = "[48;5;008m"
    red = "[48;5;009m"
    green = "[48;5;010m"
    orange = "[48;5;011m"
    blue = "[48;5;012m"
    purple = "[48;5;013m"
    cyan = "[48;5;014m"
    white = "[48;5;015m"


CodeStart = "\33"
Reset = "[0m"

if __name__ == "__main__":
    for i in [0, 38, 48]:
        for j in [5]:
            for k in range(16):
                print("\33[{};{};{}m{:02d},{},{:03d}\33[0m\t".format(i, j, k, i, j, k),
                      end='')
                if (k + 1) % 6 == 0:
                    print("")
            print("")
