from typing import TYPE_CHECKING, Type, Optional, Callable

if TYPE_CHECKING:
    from .base import Configs


class ConfigItem:
    def __init__(self, *,
                 key: str,
                 configs_class: Type['Configs'],
                 has_annotation: bool, annotation: any,
                 has_value: bool, value: any):
        self.key = key
        if annotation is None:
            annotation = type(value)
        self.annotation = annotation
        self.value = value
        self.has_annotation = has_annotation
        self.has_value = has_value
        self.configs_class = configs_class

    def update(self, k: 'ConfigItem'):
        if k.has_annotation:
            self.has_annotation = True
            self.annotation = k.annotation

        if k.has_value:
            self.has_value = True
            self.value = k.value

    # def calc(self, option: Optional[str] = None):
    #     return self.configs_class.calc(self, option)
    #
    # def __call__(self, func: Callable):
    #     return self.configs_class.calc_wrap(func, self)
