from enum import Enum
from typing import NamedTuple, Dict


class IndicatorType(Enum):
    queue = 'queue'
    histogram = 'histogram'
    scalar = 'scalar'
    pair = 'pair'


class IndicatorOptions(NamedTuple):
    is_print: bool = False
    queue_size: int = 10

    def to_dict(self) -> Dict:
        return dict(is_print=self.is_print,
                    queue_size=self.queue_size)


class Indicator(NamedTuple):
    name: str
    type_: IndicatorType
    options: IndicatorOptions

    def to_dict(self) -> Dict:
        return dict(name=self.name,
                    type=self.type_.value,
                    options=self.options.to_dict())
