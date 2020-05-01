from typing import Dict, overload, Optional

from lab.internal.logger import logger_singleton as _internal


def set_queue(name: str, queue_size=10, is_print=False):
    from lab.internal.logger.store.indicators import Queue
    _internal().add_indicator(Queue(name, queue_size, is_print))


def set_histogram(name: str, is_print=False):
    from lab.internal.logger.store.indicators import Histogram
    _internal().add_indicator(Histogram(name, is_print))


def set_scalar(name: str, is_print=False):
    from lab.internal.logger.store.indicators import Scalar
    _internal().add_indicator(Scalar(name, is_print))


def set_indexed_scalar(name: str):
    from lab.internal.logger.store.indicators import IndexedScalar
    _internal().add_indicator(IndexedScalar(name))


def set_image(name: str, is_print=False):
    from lab.internal.logger.store.artifacts import Image
    _internal().add_artifact(Image(name, is_print))


def set_text(name: str, is_print=False):
    from lab.internal.logger.store.artifacts import Text
    _internal().add_artifact(Text(name, is_print))


def set_indexed_text(name: str, title: Optional[str] = None, is_print=False):
    from lab.internal.logger.store.artifacts import IndexedText
    _internal().add_artifact(IndexedText(name, title, is_print))


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
    _internal().store(*args, **kwargs)


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
    """
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
