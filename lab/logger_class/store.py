from collections import deque
from pathlib import PurePath
from typing import Dict, List

import lab
from lab import util
from .indicators import IndicatorType, IndicatorOptions, Indicator
from .writers import Writer


class Store:
    indicators: Dict[str, Indicator]

    def __init__(self, logger: 'lab.Logger'):
        self.values = {}
        # self.queues = {}
        # self.histograms = {}
        # self.pairs: Dict[str, List[Tuple[int, int]]] = {}
        # self.scalars = {}
        self.__logger = logger
        self.indicators = {}
        self.__indicators_file = None

    def save_indicators(self, file: PurePath):
        self.__indicators_file = file

        indicators = {k: ind.to_dict() for k, ind in self.indicators.items()}
        with open(str(file), "w") as file:
            file.write(util.yaml_dump(indicators))

    def add_indicator(self, indicator: Indicator):
        """
        ### Add an indicator
        """

        assert indicator.name not in self.indicators

        self.indicators[indicator.name] = indicator

        self.__init_value(indicator.name)

        if self.__indicators_file is not None:
            self.save_indicators(self.__indicators_file)

    def __init_value(self, name):
        ind = self.indicators[name]
        if ind.type_ == 'queue':
            self.values[name] = deque(maxlen=ind.options.queue_size)
        else:
            self.values[name] = []

    def _store_list(self, items: List[Dict[str, float]]):
        for item in items:
            self.store(**item)

    def _store_kv(self, k, v):
        if k not in self.indicators:
            self.__logger.add_indicator(k, IndicatorType.scalar, IndicatorOptions(is_print=True))

        if self.indicators[k].type_ == IndicatorType.pair:
            if type(v) == tuple:
                assert len(v) == 2
                self.values[k].append((v[0], v[1]))
            else:
                assert type(v) == list
                self.values[k] += v
        else:
            self.values[k].append(v)

    def _store_kvs(self, **kwargs):
        for k, v in kwargs.items():
            self._store_kv(k, v)

    def store(self, *args, **kwargs):
        """
        ### Stores a value in the logger_class.

        This may be added to a queue, a list or stored as
        a TensorBoard histogram depending on the
        type of the indicator.
        """
        assert len(args) <= 2

        if len(args) == 0:
            self._store_kvs(**kwargs)
        elif len(args) == 1:
            assert not kwargs
            assert isinstance(args[0], list)
            self._store_list(args[0])
        elif len(args) == 2:
            assert isinstance(args[0], str)
            if isinstance(args[1], list):
                for v in args[1]:
                    self._store_kv(args[0], v)
            else:
                self._store_kv(args[0], args[1])

    def clear(self):
        for k, v in self.indicators.items():
            if v.type_ != IndicatorType.queue:
                self.__init_value(k)

    def write(self, writer: Writer, global_step):
        return writer.write(global_step=global_step,
                            values=self.values,
                            indicators=self.indicators)
