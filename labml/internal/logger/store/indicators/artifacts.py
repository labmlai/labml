from abc import ABC
from collections import OrderedDict
from typing import Dict, Optional, Any
from uuid import uuid1

from labml import logger
from labml.internal.util.values import to_numpy
from labml.logger import Text as TextStyle
from . import Indicator

try:
    import matplotlib.pyplot as plt
except (ImportError, ModuleNotFoundError):
    plt = None

try:
    import torch
except ImportError:
    torch = None


class Artifact(Indicator, ABC):
    def __init__(self, *, name: str, is_print: bool):
        super().__init__(name=name, is_print=is_print)

    def _collect_value(self, key: str, value):
        raise NotImplementedError()

    def get_print_length(self) -> Optional[int]:
        raise NotImplementedError()

    @property
    def is_indexed(self) -> bool:
        return False

    def get_string(self, key: str, others: Dict[str, 'Artifact']) -> Optional[str]:
        return None

    def print_all(self):
        pass

    def keys(self):
        raise NotImplementedError()

    def get_value(self, key: str):
        return None

    def collect_value(self, value):
        if type(value) == tuple and len(value) == 2:
            key = value[0]
            value = value[1]
        else:
            key = uuid1().hex

        self._collect_value(key, value)


class _Collection(Artifact, ABC):
    _values: Dict[str, Any]

    def __init__(self, name: str, is_print=False):
        super().__init__(name=name, is_print=is_print)
        self._values = OrderedDict()

    def _collect_value(self, key: str, value):
        self._values[key] = value

    def clear(self):
        self._values = OrderedDict()

    def is_empty(self) -> bool:
        return len(self._values) == 0

    def keys(self):
        return self._values.keys()

    def get_value(self, key: str):
        return self._values[key]


class Tensor(_Collection):
    def __init__(self, name: str, is_once: bool = False):
        super().__init__(name=name, is_print=False)
        self.is_once = is_once

    def to_dict(self) -> Dict:
        return dict(class_name=self.__class__.__name__,
                    name=self.name,
                    is_print=self.is_print,
                    is_once=self.is_once)

    def get_print_length(self) -> Optional[int]:
        return None

    def copy(self, key: str):
        return Tensor(key, is_once=self.is_once)

    def _collect_value(self, key: str, value):
        self._values[key] = to_numpy(value)

    def equals(self, value: any) -> bool:
        if not isinstance(value, Tensor):
            return False
        return value.name == self.name and value.is_once == self.is_once


class Image(_Collection):
    def get_print_length(self) -> Optional[int]:
        return None

    def print_all(self):
        if plt is None:
            logger.log(('matplotlib', logger.Text.highlight),
                       ' not found. So cannot display images')
        images = [to_numpy(v) for v in self._values.values()]
        cols = 3
        fig: plt.Figure
        fig, axs = plt.subplots((len(images) + cols - 1) // cols, cols,
                                sharex='all', sharey='all',
                                figsize=(8, 10))
        fig.suptitle(self.name)
        for i, img in enumerate(images):
            ax: plt.Axes = axs[i // cols, i % cols]
            ax.imshow(img)
        plt.show()

    def copy(self, key: str):
        return Image(key, is_print=self.is_print)


class Text(_Collection):
    def get_print_length(self) -> Optional[int]:
        return None

    def print_all(self):
        logger.log(self.name, TextStyle.heading)
        for t in self._values.values():
            logger.log(t, TextStyle.value)

    def copy(self, key: str):
        return Text(key, is_print=self.is_print)

    def equals(self, value: any) -> bool:
        if not isinstance(value, Text):
            return False
        return value.name == self.name and value.is_print == self.is_print


class IndexedText(_Collection):
    def __init__(self, name: str, title: Optional[str] = None, is_print=False):
        super().__init__(name=name, is_print=is_print)
        self.title = title

    @_Collection.is_indexed.getter
    def is_indexed(self) -> bool:
        return True

    def get_print_length(self) -> Optional[int]:
        return max((len(v) for v in self._values.values()))

    def get_string(self, key: str, others: Dict[str, Artifact]) -> Optional[str]:
        return self._values[key]

    def copy(self, key: str):
        return IndexedText(key, title=self.title, is_print=self.is_print)

    def equals(self, value: any) -> bool:
        if not super().equals(value):
            return False
        return value.title == self.title
