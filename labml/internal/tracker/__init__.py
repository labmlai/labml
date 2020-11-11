from pathlib import PurePath
from typing import Dict, List, Optional, Callable, Union, Tuple

from labml.internal import util
from labml.internal.lab import lab_singleton, LabYamlNotfoundError
from labml.internal.util import strings
from .indicators import Indicator
from .indicators.factory import load_indicator_from_dict, create_default_indicator
from .indicators.numeric import Scalar
from .namespace import Namespace
from .writers import Writer
from .writers.screen import ScreenWriter
from ..logger import LogPart
from labml.internal.util.colors import StyleCode
from ... import logger
from ...logger import Text


class Tracker:
    __loop_counter: int
    __set_looping_indicators: Optional[Callable[[List[Union[str, Tuple[str, Optional[StyleCode]]]]], None]]

    dot_indicators: Dict[str, Indicator]
    namespaces: List[Namespace]
    indicators: Dict[str, Indicator]

    def __init__(self):
        self.__store = None
        self.__writers: List[Writer] = []

        self.__start_global_step: Optional[int] = None
        self.__global_step: Optional[int] = None
        self.__last_global_step: Optional[int] = None

        self.__set_looping_indicators = None
        self.__loop_counter = 0

        self.indicators = {}
        self.dot_indicators = {}
        self.__indicators_file = None
        self.namespaces = []
        self.is_indicators_updated = True
        self.reset_store()

    def __assert_name(self, name: str, value: any):
        if name.endswith("."):
            if name in self.dot_indicators:
                assert self.dot_indicators[name].equals(value)

        assert name not in self.indicators, f"{name} already used"

    def add_writer(self, writer: Writer):
        self.__writers.append(writer)

    def reset_writers(self):
        self.__writers = []

    def write_h_parameters(self, hparams: Dict[str, any]):
        for w in self.__writers:
            w.write_h_parameters(hparams)

    def _write_writer(self, writer: Writer, global_step):
        return writer.write(global_step=global_step,
                            indicators=self.indicators)

    def clear(self):
        for k, v in self.indicators.items():
            v.clear()

    def write(self):
        global_step = self.global_step

        self.save_indicators()

        indicators_print = None

        for w in self.__writers:
            if isinstance(w, ScreenWriter):
                indicators_print = self._write_writer(w, global_step)
            else:
                self._write_writer(w, global_step)
        self.clear()

        if indicators_print is not None:
            if self.__is_looping:
                self.__set_looping_indicators(indicators_print)
            else:
                parts = [(f"{self.global_step :8,}:  ", Text.highlight)]
                parts += indicators_print
                logger.log(parts, is_new_line=False)

    @property
    def global_step(self) -> int:
        if self.__global_step is not None:
            return self.__global_step

        global_step = 0
        if self.__start_global_step is not None:
            global_step = self.__start_global_step

        if self.__is_looping:
            return global_step + self.__loop_counter

        if self.__last_global_step is not None:
            return self.__last_global_step

        return global_step

    def reset_store(self):
        self.indicators = {}
        self.dot_indicators = {}
        self.__indicators_file = None
        self.namespaces = []
        self.is_indicators_updated = True
        try:
            for ind in lab_singleton().indicators:
                self.add_indicator(load_indicator_from_dict(ind))
        except LabYamlNotfoundError:
            pass

    def add_indicator(self, indicator: Indicator):
        self.dot_indicators[indicator.name] = indicator
        self.is_indicators_updated = True

    def save_indicators(self, file: Optional[PurePath] = None):
        if not self.is_indicators_updated:
            return
        self.is_indicators_updated = False
        if file is None:
            if self.__indicators_file is None:
                return
            file = self.__indicators_file
        else:
            self.__indicators_file = file

        wildcards = {k: ind.to_dict() for k, ind in self.dot_indicators.items()}
        inds = {k: ind.to_dict() for k, ind in self.indicators.items()}
        with open(str(file), "w") as file:
            file.write(util.yaml_dump({'wildcards': wildcards,
                                       'indicators': inds}))

        for w in self.__writers:
            w.save_indicators(self.dot_indicators, self.indicators)

    def _create_indicator(self, key: str, value: any):
        if key in self.indicators:
            return

        ind_key, ind_score = strings.find_best_pattern(key, self.dot_indicators.keys())
        if ind_key is None:
            raise ValueError(f"Cannot find matching indicator for {key}")
        if ind_score == 0:
            is_print = self.dot_indicators[ind_key].is_print
            self.indicators[key] = create_default_indicator(key, value, is_print)
        else:
            self.indicators[key] = self.dot_indicators[ind_key].copy(key)
        self.is_indicators_updated = True

    def store(self, key: str, value: any):
        if key.endswith('.'):
            key = '.'.join([key[:-1]] + [ns.name for ns in self.namespaces])

        self._create_indicator(key, value)
        self.indicators[key].collect_value(value)

    def new_line(self):
        for w in self.__writers:
            if isinstance(w, ScreenWriter):
                logger.log()

    def namespace(self, name: str):
        return Namespace(tracker=self, name=name)

    def namespace_enter(self, ns: Namespace):
        self.namespaces.append(ns)

    def namespace_exit(self, ns: Namespace):
        if len(self.namespaces) == 0:
            raise RuntimeError("Impossible")

        if ns is not self.namespaces[-1]:
            raise RuntimeError("Impossible")

        self.namespaces.pop(-1)

    def set_global_step(self, global_step: Optional[int]):
        self.__global_step = global_step

    def set_start_global_step(self, global_step: Optional[int]):
        self.__start_global_step = global_step

    def add_global_step(self, increment_global_step: int = 1):
        if self.__global_step is None:
            self.__global_step = self.global_step

        self.__global_step += increment_global_step

    @property
    def __is_looping(self):
        return self.__set_looping_indicators is not None

    def start_loop(self, set_looping_indicators: Callable[[List[LogPart]], None]):
        self.__set_looping_indicators = set_looping_indicators

    def loop_count(self, value: int):
        self.__loop_counter = value

    def finish_loop(self):
        self.__last_global_step = self.global_step
        self.__set_looping_indicators = None
        for w in self.__writers:
            w.flush()


_internal: Optional[Tracker] = None


def tracker_singleton() -> Tracker:
    global _internal
    if _internal is None:
        _internal = Tracker()

    return _internal
