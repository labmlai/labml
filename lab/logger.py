from typing import Union, List, Tuple, Optional, overload, Dict

from lab.internal.logger.colors import StyleCode


class Style(StyleCode):
    r"""
    Output styles
    """

    none = None
    normal = 'normal'
    bold = 'bold'
    underline = 'underline'
    light = 'light'


class Color(StyleCode):
    r"""
    Output colors
    """

    none = None
    black = 'black'
    red = 'red'
    green = 'green'
    orange = 'orange'
    blue = 'blue'
    purple = 'purple'
    cyan = 'cyan'
    white = 'white'


class Text(StyleCode):
    r"""
    Standard styles we use in lab
    """

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


@overload
def log():
    ...


@overload
def log(message: str, *, is_new_line=True):
    ...


@overload
def log(message: str, color: StyleCode,
        *,
        is_new_line=True):
    ...


@overload
def log(message: str, colors: List[StyleCode],
        *,
        is_new_line=True):
    ...


@overload
def log(messages: List[Union[str, Tuple[str, StyleCode]]],
        *,
        is_new_line=True):
    ...


def log(message: Optional[Union[str, List[Union[str, Tuple[str, StyleCode]]]]] = None,
        color: List[StyleCode] or StyleCode or None = None,
        *,
        is_new_line=True):
    r"""
    This has multiple overloads

    .. function:: log(message: str, *, is_new_line=True)
        :noindex:

    .. function:: log(message: str, color: StyleCode, *, is_new_line=True)
        :noindex:

    .. function:: log(message: str, colors: List[StyleCode], *, is_new_line=True)
        :noindex:

    .. function:: log(messages: List[Union[str, Tuple[str, StyleCode]]], *, is_new_line=True)
        :noindex:

    Arguments:
        message (str): string to be printed
        color (StyleCode): color/style of the message
        colors (List[StyleCode]): list of colors/styles for the message
        messages (List[Union[str, Tuple[str, StyleCode]]]): a list of messages.
            Each element should be either a string or a tuple of string and styles.

    Keyword Arguments:
        is_new_line (bool): whether to print a new line at the end

    Example::
        >>> logger.log("test")
    """
    from lab.internal.logger import logger_singleton as _internal

    if message is None:
        assert is_new_line == True
        _internal().log('', is_new_line=True)
    if type(message) == str:
        _internal().log([(message, color)], is_new_line=is_new_line)
    elif type(message) == list:
        _internal().log(message, is_new_line=is_new_line)


@overload
def inspect(items: Dict):
    ...


@overload
def inspect(items: List):
    ...


@overload
def inspect(*items: any):
    ...


@overload
def inspect(**items: any):
    ...


def inspect(*args, **kwargs):
    """
    🎨 Pretty prints a set of values.
    """

    from lab.internal.logger import logger_singleton as _internal

    _internal().info(*args, **kwargs)
