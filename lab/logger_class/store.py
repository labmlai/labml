from collections import deque
from typing import Dict, List, Tuple

import lab
from lab.logger_class.writers import Writer


class Store:
    def __init__(self, logger: 'lab.Logger'):
        self.queues = {}
        self.histograms = {}
        self.pairs: Dict[str, List[Tuple[int, int]]] = {}
        self.scalars = {}
        self.__logger = logger

    def add_indicator(self, name: str, *,
                      queue_limit: int = None,
                      is_histogram: bool = True,
                      is_pair: bool = False):
        """
        ### Add an indicator
        """

        if queue_limit is not None:
            self.queues[name] = deque(maxlen=queue_limit)
        elif is_histogram:
            self.histograms[name] = []
        else:
            self.scalars[name] = []

        if is_pair:
            self.pairs[name] = []

    def _store_list(self, items: List[Dict[str, float]]):
        for item in items:
            self.store(**item)

    def _store_kv(self, k, v):
        if k in self.queues:
            self.queues[k].append(v)
        elif k in self.histograms:
            self.histograms[k].append(v)
        elif k in self.pairs:
            if type(v) == tuple:
                assert len(v) == 2
                self.pairs[k].append((v[0], v[1]))
            else:
                assert type(v) == list
                self.pairs[k] += v
        else:
            if k not in self.scalars:
                self.__logger.add_indicator(k, is_histogram=False, is_print=True)
            self.scalars[k].append(v)

    def _store_kvs(self, **kwargs):
        for k, v in kwargs.items():
            self._store_kv(k, v)

    def has_key(self, k):
        if k in self.queues:
            return len(self.queues[k]) > 0
        elif k in self.histograms:
            return len(self.histograms[k]) > 0
        elif k in self.pairs:
            return len(self.pairs[k]) > 0
        else:
            return len(self.scalars[k]) > 0

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
        for k in self.histograms:
            self.histograms[k] = []
        for k in self.scalars:
            self.scalars[k] = []
        for k in self.pairs:
            self.pairs[k] = []

    def write(self, writer: Writer, global_step):
        return writer.write(global_step=global_step,
                            queues=self.queues,
                            histograms=self.histograms,
                            pairs=self.pairs,
                            scalars=self.scalars)
