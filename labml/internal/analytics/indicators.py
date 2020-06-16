from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple, Dict, Union

from labml import lab
from labml.internal import util
from labml.internal.experiment.experiment_run import RunInfo
from labml.internal.util.strings import is_pattern_match


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
                 props: Dict[str, any], *,
                 select: Optional[StepSelect] = None,
                 is_mean: bool = False):
        self.uuid = uuid
        self.class_ = class_
        self.key = key
        self.props = props
        if class_ == IndicatorClass.tensor and props.get('is_once', False):
            select = None
        if select is None:
            select = StepSelect(None, None)
        self.select = select
        self.is_mean = is_mean

    def hash_str(self):
        return f"{self.uuid}#{self.key}"

    @property
    def is_distribution(self):
        return self.class_ in [IndicatorClass.histogram, IndicatorClass.queue]

    @property
    def is_scalar(self):
        return self.class_ == IndicatorClass.scalar


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

    def __getitem__(self, item: Union[slice, str]):
        if isinstance(item, slice):
            select = StepSelect(item.start, item.stop)
            inds = [Indicator(ind.key, ind.class_, ind.uuid, ind.props,
                              select=select, is_mean=ind.is_mean)
                    for ind in self._indicators]
            return IndicatorCollection(inds)
        elif isinstance(item, str):
            inds = [ind for ind in self._indicators if is_pattern_match(ind.key, item)]
            return IndicatorCollection(inds)
        else:
            raise ValueError(item)

    def mean(self):
        inds = [Indicator(ind.key, ind.class_, ind.uuid, ind.props,
                          select=ind.select,
                          is_mean=True) for ind in self._indicators]
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

        if 'indicators' not in indicators:
            raise RuntimeError("This run is corrupted or from an old version of LabML. "
                               "Please update labml_dashboard and run it on this project. "
                               "It will automatically migrate all the experiments.")

        indicators = indicators['indicators']

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
            elif cn == 'Tensor':
                class_ = IndicatorClass.tensor

            if class_ is None:
                continue
            inds.append(Indicator(k, class_, self.run_info.uuid, v))

        self.indicators = IndicatorCollection(inds)
