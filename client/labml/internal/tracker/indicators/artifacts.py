from abc import ABC
from collections import OrderedDict
from typing import Dict, Optional, Any
from uuid import uuid1

import numpy as np

from labml.internal.util.values import to_numpy
from . import Indicator


class Artifact(Indicator, ABC):
    def __init__(self, *, name: str, is_print: bool, options: Optional[Dict]):
        super().__init__(name=name, is_print=is_print, options=options)

    def _collect_value(self, key: str, value):
        raise NotImplementedError()

    @property
    def is_indexed(self) -> bool:
        return False

    def keys(self):
        raise NotImplementedError()

    def get_value(self, key: str):
        return None

    def get_values(self):
        return None

    def collect_value(self, value):
        if type(value) == tuple and len(value) == 2:
            key = value[0]
            value = value[1]
        else:
            key = uuid1().hex

        self._collect_value(key, value)


class _Collection(Artifact, ABC):
    _last_add_step: int
    _values: Dict[str, Any]

    def __init__(self, name: str, is_print: bool, density: Optional[float] = None, options: Optional[Dict] = None):
        super().__init__(name=name, is_print=is_print, options=options)
        self._values = OrderedDict()
        self._density = density
        self._last_add_step = -1

    def _collect_value(self, key: str, value):
        from labml import tracker
        if self._density is None:
            self._values[key] = value
        else:
            steps = tracker.get_global_step() - self._last_add_step
            steps *= self._density
            if np.random.uniform() < 1 - 0.99 ** steps:
                self._values[key] = value

        self._last_add_step = tracker.get_global_step()

    def clear(self):
        self._values = OrderedDict()

    def is_empty(self) -> bool:
        return len(self._values) == 0

    def keys(self):
        return self._values.keys()

    def get_value(self, key: str):
        return self._values[key]

    def get_values(self):
        return self._values


class Tensor(_Collection):
    def __init__(self, name: str, is_once: bool = False, options: Optional[Dict] = None):
        super().__init__(name=name, is_print=False, density=None, options=options)
        self.is_once = is_once

    def to_dict(self) -> Dict:
        return dict(class_name=self.__class__.__name__,
                    name=self.name,
                    is_print=self.is_print,
                    is_once=self.is_once)

    def copy(self, key: str):
        return Tensor(key, is_once=self.is_once, options=self.options)

    def _collect_value(self, key: str, value):
        self._values[key] = to_numpy(value)

    def equals(self, value: any) -> bool:
        if not isinstance(value, Tensor):
            return False
        return value.name == self.name and value.is_once == self.is_once


class Image(_Collection):
    def copy(self, key: str):
        return Image(key, is_print=self.is_print, density=self._density, options=self.options)

    def get_images(self):
        images = [to_numpy(v) for v in self.get_values().values()]
        images = np.concatenate(images)

        if images.dtype.type in (np.float32, np.float64):
            images = np.clip(images, 0., 1.)
        else:
            images = np.clip(images, 0, 255)

        return images


class Text(_Collection):
    def copy(self, key: str):
        return Text(key, is_print=self.is_print, density=self._density, options=self.options)

    def equals(self, value: any) -> bool:
        if not isinstance(value, Text):
            return False
        return value.name == self.name and value.is_print == self.is_print


class IndexedText(_Collection):
    def __init__(self, name: str, title: Optional[str] = None, is_print: bool = False, options: Optional[Dict] = None):
        super().__init__(name=name, is_print=is_print, density=None, options=options)
        self.title = title

    @_Collection.is_indexed.getter
    def is_indexed(self) -> bool:
        return True

    def copy(self, key: str):
        return IndexedText(key, title=self.title, is_print=self.is_print, options=self.options)

    def equals(self, value: any) -> bool:
        if not super().equals(value):
            return False
        return value.title == self.title
