from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional

from labml import lab
from labml.internal import util
from labml.internal.experiment import experiment_run


class RunsSet:
    def __init__(self):
        experiment_path = Path(lab.get_experiments_path())
        runs = {}
        for exp_path in experiment_path.iterdir():
            for run_path in exp_path.iterdir():
                runs[run_path.name] = run_path

        self.runs = runs

    def get(self, uuid: str):
        run_path = self.runs[uuid]
        run_info_path = run_path / 'run.yaml'

        with open(str(run_info_path), 'r') as f:
            data = util.yaml_load(f.read())
            run = experiment_run.RunInfo.from_dict(run_path.parent, data)

        return run


class IndicatorClass(Enum):
    scalar = 'scalar'
    histogram = 'histogram'
    queue = 'queue'


class Indicator:
    def __init__(self, key: str, class_: IndicatorClass, uuid: str):
        self.uuid = uuid
        self.class_ = class_
        self.key = key

    def hash_str(self):
        return f"{self.uuid}#{self.key}"


class IndicatorCollection:
    _indicators: List[Indicator]

    def __init__(self, indicators: List[Indicator]):
        has = set()
        self._indicators = []
        for ind in indicators:
            h = ind.hash_str()
            if h in has:
                continue
            has.add(h)
            self._indicators.append(ind)

        self._indicator_keys = {ind.key.replace('.', '_'): ind.key for ind in self._indicators}
        self._indicators_list = [k for k in self._indicator_keys.keys()]

    def __dir__(self):
        return self._indicators_list

    def __getattr__(self, k: str):
        key = self._indicator_keys[k]
        inds = []
        for ind in self._indicators:
            if ind.key == key:
                inds.append(ind)

        return IndicatorCollection(inds)

    def __add__(self, other: 'IndicatorCollection'):
        return IndicatorCollection(self._indicators + other._indicators)

    def __radd__(self, other: Optional['IndicatorCollection']):
        if other is None:
            return IndicatorCollection(self._indicators)
        else:
            return IndicatorCollection(self._indicators + other._indicators)

    def __iter__(self):
        return iter(self._indicators)

    def __len__(self):
        return len(self._indicators)


class Run:
    def __init__(self, uuid: str):
        runs = RunsSet()
        self.run_info = runs.get(uuid)

        with open(str(self.run_info.indicators_path), 'r') as f:
            indicators = util.yaml_load(f.read())

        inds = []
        for k, v in indicators.items():
            cn = v['class_name']
            class_ = None
            if cn == 'Histogram':
                class_ = IndicatorClass.histogram
            elif cn == 'Queue':
                class_ = IndicatorClass.queue
            elif cn == 'IndexedScalar':
                class_ = IndicatorClass.scalar
            elif cn == 'Scalar':
                class_ = IndicatorClass.scalar

            if class_ is None:
                continue
            inds.append(Indicator(k, class_, self.run_info.uuid))

        self.indicators = IndicatorCollection(inds)
