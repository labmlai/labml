from typing import Union, List, Tuple, overload, Dict

from labml.internal.util.colors import StyleCode


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
    Standard styles we use in labml
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
def log(message: str, *, is_new_line: bool = True):
    ...


@overload
def log(message: str, color: StyleCode,
        *,
        is_new_line: bool = True):
    ...


@overload
def log(message: str, colors: List[StyleCode],
        *,
        is_new_line: bool = True):
    ...


@overload
def log(messages: List[Union[str, Tuple[str, StyleCode]]],
        *,
        is_new_line: bool = True):
    ...


@overload
def log(*args: Union[str, Tuple[str, StyleCode]],
        is_new_line: bool = True):
    ...


def log(*args, is_new_line: bool = True):
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

    .. function:: log(*args: Union[str, Tuple[str, StyleCode]], is_new_line: bool = True)
        :noindex:

    Arguments:
        message (str): string to be printed
        color (StyleCode): color/style of the message
        colors (List[StyleCode]): list of colors/styles for the message
        args (Union[str, Tuple[str, StyleCode]]): list of messages
            Each element should be either a string or a tuple of string and styles.
        messages (List[Union[str, Tuple[str, StyleCode]]]): a list of messages.
            Each element should be either a string or a tuple of string and styles.

    Keyword Arguments:
        is_new_line (bool): whether to print a new line at the end

    Example::
        >>> logger.log("test")
    """
    from labml.internal.logger import logger_singleton as _internal

    if len(args) == 0:
        assert is_new_line == True
        _internal().log([], is_new_line=True)
    elif len(args) == 1:
        message = args[0]
        if isinstance(message, str):
            _internal().log([(message, None)], is_new_line=is_new_line)
        elif type(message) == list:
            _internal().log(message, is_new_line=is_new_line)
        else:
            assert False
    elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], StyleCode):
        _internal().log([(args[0], args[1])], is_new_line=is_new_line)
    else:
        _internal().log(args, is_new_line=is_new_line)


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
    Pretty prints a set of values.

    This has multiple overloads

    .. function:: inspect(items: Dict)
        :noindex:

    .. function:: inspect(items: List)
        :noindex:

    .. function:: inspect(*items: any)
        :noindex:
    """

    from labml.internal.logger import logger_singleton as _internal

    _internal().info(*args, **kwargs)
