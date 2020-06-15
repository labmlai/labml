from typing import Dict, overload, Optional

from labml.internal.logger import logger_singleton as _internal


def set_global_step(global_step: Optional[int]):
    _internal().set_global_step(global_step)


def add_global_step(increment_global_step: int = 1):
    _internal().add_global_step(int(increment_global_step))


def get_global_step() -> int:
    return _internal().global_step


def set_queue(name: str, queue_size: int = 10, is_print: bool = False):
    from labml.internal.logger.store.indicators.numeric import Queue
    _internal().add_indicator(Queue(name, queue_size, is_print))


def set_histogram(name: str, is_print: bool = False):
    from labml.internal.logger.store.indicators.numeric import Histogram
    _internal().add_indicator(Histogram(name, is_print))


def set_scalar(name: str, is_print: bool = False):
    from labml.internal.logger.store.indicators.numeric import Scalar
    _internal().add_indicator(Scalar(name, is_print))


def set_indexed_scalar(name: str):
    from labml.internal.logger.store.indicators.indexed import IndexedScalar
    _internal().add_indicator(IndexedScalar(name))


def set_image(name: str, is_print: bool = False):
    from labml.internal.logger.store.indicators.artifacts import Image
    _internal().add_indicator(Image(name, is_print))


def set_text(name: str, is_print: bool = False):
    from labml.internal.logger.store.indicators.artifacts import Text
    _internal().add_indicator(Text(name, is_print))


def set_tensor(name: str, is_once: bool = False):
    from labml.internal.logger.store.indicators.artifacts import Tensor
    _internal().add_indicator(Tensor(name, is_once=is_once))


def set_indexed_text(name: str, title: Optional[str] = None, is_print: bool = False):
    from labml.internal.logger.store.indicators.artifacts import IndexedText
    _internal().add_indicator(IndexedText(name, title, is_print))


def _add_dict(values: Dict[str, any]):
    for k, v in values.items():
        _internal().store(k, v)


@overload
def add(values: Dict[str, any]):
    ...


@overload
def add(name: str, value: any):
    ...


@overload
def add(**kwargs: any):
    ...


def add(*args, **kwargs):
    """
    This has multiple overloads

    .. function:: add(values: Dict[str, any])
        :noindex:

    .. function:: add(name: str, value: any)
        :noindex:

    .. function:: add(**kwargs: any)
        :noindex:
    """
    assert len(args) <= 2

    if len(args) == 0:
        _add_dict(kwargs)
    elif len(args) == 1:
        assert not kwargs
        assert isinstance(args[0], dict)
        _add_dict(args[0])
    elif len(args) == 2:
        assert not kwargs
        assert isinstance(args[0], str)
        _internal().store(args[0], args[1])


@overload
def save():
    ...


@overload
def save(global_step: int):
    ...


@overload
def save(values: Dict[str, any]):
    ...


@overload
def save(name: str, value: any):
    ...


@overload
def save(**kwargs: any):
    ...


@overload
def save(global_step: int, values: Dict[str, any]):
    ...


@overload
def save(global_step: int, name: str, value: any):
    ...


@overload
def save(global_step: int, **kwargs: any):
    ...


def save(*args, **kwargs):
    r"""
    This has multiple overloads

    .. function:: save()
        :noindex:

    .. function:: save(global_step: int)
        :noindex:

    .. function:: save(values: Dict[str, any])
        :noindex:

    .. function:: save(name: str, value: any)
        :noindex:

    .. function:: save(**kwargs: any)
        :noindex:

    .. function:: save(global_step: int, values: Dict[str, any])
        :noindex:

    .. function:: save(global_step: int, name: str, value: any)
        :noindex:

    .. function:: save(global_step: int, **kwargs: any)
        :noindex:
    """
    if len(args) > 0 and type(args[0]) == int:
        _internal().set_global_step(args[0])
        args = args[1:]

    if len(args) > 0 or len(kwargs.keys()) > 0:
        add(*args, **kwargs)

    _internal().write()


def namespace(name: str):
    r"""
    Set a namespace for tracking
    """
    return _internal().store_namespace(name)


def reset():
    r"""
    Reset indicators, for a new experiment
    """
    _internal().reset_store()
