from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple, Dict

from labml import lab
from labml.internal import util
from labml.internal.experiment.experiment_run import RunInfo


class RunsSet:
    _runs: Dict[str, Tuple[Path, str]]

    def __init__(self):
        experiment_path = Path(lab.get_experiments_path())
        runs = {}
        for exp_path in experiment_path.iterdir():
            for run_path in exp_path.iterdir():
                runs[run_path.name] = (run_path, experiment_path.name)

        self._runs = runs

    def get(self, uuid: str) -> Tuple[RunInfo, str]:
        run_path = self._runs[uuid][0]
        run_info_path = run_path / 'run.yaml'

        with open(str(run_info_path), 'r') as f:
            data = util.yaml_load(f.read())
            run = RunInfo.from_dict(run_path.parent, data)

        return run, self._runs[uuid][1]


class IndicatorClass(Enum):
    scalar = 'scalar'
    histogram = 'histogram'
    queue = 'queue'
    tensor = 'tensor'


class StepSelect(NamedTuple):
    start: Optional[int]
    end: Optional[int]


class Indicator:
    def __init__(self, key: str, class_: IndicatorClass, uuid: str,
                 props: Dict[str, any],
                 select: Optional[StepSelect]):
        self.uuid = uuid
        self.class_ = class_
        self.key = key
        self.props = props
        if class_ == IndicatorClass.tensor and props.get('is_once', False):
            select = None
        if select is None:
            select = StepSelect(None, None)
        self.select = select

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

    def __getitem__(self, item: slice):
        select = StepSelect(item.start, item.stop)
        inds = [Indicator(ind.key, ind.class_, ind.uuid, ind.props, select)
                for ind in self._indicators]
        return IndicatorCollection(inds)


class Run:
    indicators: IndicatorCollection
    name: str
    run_info: RunInfo

    def __init__(self, uuid: str):
        runs = RunsSet()
        self.run_info, self.name = runs.get(uuid)

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
            inds.append(Indicator(k, class_, self.run_info.uuid, v, None))

        with open(str(self.run_info.artifacts_path), 'r') as f:
            artifacts = util.yaml_load(f.read())

        for k, v in artifacts.items():
            cn = v['class_name']
            class_ = None
            if cn == 'Tensor':
                class_ = IndicatorClass.tensor

            if class_ is None:
                continue
            inds.append(Indicator(k, class_, self.run_info.uuid, v, None))

        self.indicators = IndicatorCollection(inds)
