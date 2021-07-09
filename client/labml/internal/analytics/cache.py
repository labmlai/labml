from pathlib import PurePath
from typing import List, Tuple

import numpy as np

from labml.internal.analytics.indicators import IndicatorClass, Indicator, Run, IndicatorCollection, \
    StepSelect
from labml.internal.analytics.sqlite import SQLiteAnalytics
from labml.internal.analytics.tensorboard import TensorBoardAnalytics

_RUNS = {}

_TENSORBOARD = {}

_SQLITE = {}

_NUMPY_ARRAYS = {}


def get_run(uuid: str) -> Run:
    if uuid not in _RUNS:
        _RUNS[uuid] = Run(uuid)

    return _RUNS[uuid]


def get_name(ind: Indicator) -> List[str]:
    run = get_run(ind.uuid)
    return [run.name, ind.uuid[-5:], run.run_info.comment, ind.key]


def get_tensorboard_data(indicator: Indicator):
    run = get_run(indicator.uuid)

    if indicator.uuid not in _TENSORBOARD:
        _TENSORBOARD[indicator.uuid] = TensorBoardAnalytics(run.run_info.tensorboard_log_path)

    tb: TensorBoardAnalytics = _TENSORBOARD[indicator.uuid]
    try:
        tb.load()
    except FileNotFoundError:
        return None

    key = indicator.key
    if indicator.is_distribution and indicator.is_mean:
        key = f'{key}.mean'

    try:
        tensor = tb.tensor(key, indicator.select.start, indicator.select.end)
    except KeyError:
        return None

    if indicator.is_distribution and not indicator.is_mean:
        data = tb.summarize_compressed_histogram(tensor)
    else:
        data = tb.summarize_scalars(tensor)

    return data


def _get_sqlite_scalar_data(sqlite: SQLiteAnalytics, key: str, select: StepSelect):
    data = sqlite.scalar(key, select.start, select.end)
    if not data:
        return None

    data = sqlite.summarize_scalars(data)

    return data


def get_sqlite_data(indicator: Indicator):
    run = get_run(indicator.uuid)

    if indicator.uuid not in _SQLITE:
        _SQLITE[indicator.uuid] = SQLiteAnalytics(run.run_info.sqlite_path)

    sqlite: SQLiteAnalytics = _SQLITE[indicator.uuid]

    if indicator.is_distribution:
        return _get_sqlite_scalar_data(sqlite, f"{indicator.key}.mean", indicator.select)
    elif indicator.is_scalar:
        return _get_sqlite_scalar_data(sqlite, indicator.key, indicator.select)
    else:
        return None


_PREFERRED_DB = 'tensorboard'


def set_preferred_db(db: str):
    global _PREFERRED_DB
    _PREFERRED_DB = db


def get_indicator_data(indicator: Indicator):
    if _PREFERRED_DB == 'tensorboard':
        data = get_tensorboard_data(indicator)
        if data is None:
            data = get_sqlite_data(indicator)
    else:
        data = get_sqlite_data(indicator)
        if data is None:
            data = get_tensorboard_data(indicator)

    return data


def get_indicators_data(indicators: IndicatorCollection):
    series = []
    names = []
    for i, ind in enumerate(indicators):
        d = get_indicator_data(ind)
        if d is not None:
            series.append(d)
            names.append(get_name(ind))

    return series, names


def _get_numpy_array(path: PurePath):
    path = str(path)
    if path not in _NUMPY_ARRAYS:
        _NUMPY_ARRAYS[path] = np.load(path)

    return _NUMPY_ARRAYS[path]


def get_artifact_files(indicator: Indicator):
    run = get_run(indicator.uuid)

    if indicator.uuid not in _SQLITE:
        _SQLITE[indicator.uuid] = SQLiteAnalytics(run.run_info.sqlite_path)

    sqlite: SQLiteAnalytics = _SQLITE[indicator.uuid]

    if indicator.class_ != IndicatorClass.tensor:
        return None

    data = sqlite.tensor(indicator.key, indicator.select.start, indicator.select.end)

    if not data:
        return None

    return data


def _get_condensed_steps(series: List[List[Tuple[int, any]]], limit: int):
    steps = {}
    step_lookups = []

    series = [s for s in series if len(s) != 1 or s[0][0] != -1]

    for s in series:
        lookup = {}
        for i, c in enumerate(s):
            steps[c[0]] = steps.get(c[0], 0) + 1
            lookup[c[0]] = i
        step_lookups.append(lookup)

    steps = [k for k, v in steps.items() if v == len(series)]
    if len(steps) == 0:
        return [], step_lookups

    steps = sorted(steps)
    last = steps[-1]
    interval = max(1, last // (limit - 1))

    condensed = [steps[0]]
    for s in steps[1:]:
        if s - condensed[-1] >= interval:
            condensed.append(s)
    if condensed[-1] != steps[-1]:
        condensed.append(steps[-1])
    return condensed, step_lookups


def get_artifacts_data(indicators: IndicatorCollection, limit: int = 100):
    series = []
    names = []
    series_inds = []
    for i, ind in enumerate(indicators):
        d = get_artifact_files(ind)
        if d is not None:
            series.append(d)
            series_inds.append(ind)
            names.append(get_name(ind))

    steps, step_lookups = _get_condensed_steps(series, limit)

    data_series = []
    for si, s in enumerate(series):
        ind: Indicator = series_inds[si]
        run: Run = get_run(ind.uuid)
        if ind.props.get('is_once', False):
            assert s[0][0] == -1
            filename = s[0][1]
            data = _get_numpy_array(run.run_info.artifacts_folder / filename)
        else:
            data = []
            for step in steps:
                filename = s[step_lookups[si][step]][1]
                data.append((step, _get_numpy_array(run.run_info.artifacts_folder / filename)))
        data_series.append(data)

    return data_series, names
