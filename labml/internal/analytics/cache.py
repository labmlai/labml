from pathlib import PurePath

import numpy as np

from labml.internal.analytics.indicators import IndicatorClass, Indicator, Run, IndicatorCollection
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


def get_tensorboard_data(indicator: Indicator):
    run = get_run(indicator.uuid)

    if indicator.uuid not in _TENSORBOARD:
        _TENSORBOARD[indicator.uuid] = TensorBoardAnalytics(run.run_info.tensorboard_log_path)

    tb: TensorBoardAnalytics = _TENSORBOARD[indicator.uuid]
    try:
        tb.load()
    except FileNotFoundError:
        return None

    try:
        tensor = tb.tensor(indicator.key)
    except KeyError:
        return None
    if indicator.class_ in [IndicatorClass.histogram, IndicatorClass.queue]:
        data = tb.summarize_compressed_histogram(tensor)
    else:
        data = tb.summarize_scalars(tensor)

    return data


def _get_sqlite_scalar_data(sqlite: SQLiteAnalytics, key: str):
    data = sqlite.scalar(key)
    if not data:
        return None

    data = sqlite.summarize_scalars(data)

    return data


def get_sqlite_data(indicator: Indicator):
    run = get_run(indicator.uuid)

    if indicator.uuid not in _SQLITE:
        _SQLITE[indicator.uuid] = SQLiteAnalytics(run.run_info.sqlite_path)

    sqlite: SQLiteAnalytics = _SQLITE[indicator.uuid]

    if indicator.class_ in [IndicatorClass.histogram, IndicatorClass.queue]:
        return _get_sqlite_scalar_data(sqlite, f"{indicator.key}.mean")
    elif indicator.class_ == IndicatorClass.scalar:
        return _get_sqlite_scalar_data(sqlite, indicator.key)
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
            names.append(ind.key)

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

    data = sqlite.tensor(indicator.key)

    if not data:
        return None

    return data


def get_artifact_data(indicator: Indicator):
    data = get_artifact_files(indicator)
    if data is None:
        return data

    run = get_run(indicator.uuid)
    data = [(c[0], _get_numpy_array(run.run_info.artifacts_folder / c[1])) for c in data]

    return data


def get_artifacts_data(indicators: IndicatorCollection, limit: int = 100):
    series = []
    names = []
    series_inds = []
    for i, ind in enumerate(indicators):
        d = get_artifact_files(ind)
        if d is not None:
            series.append(d)
            series_inds.append(ind)
            names.append(ind.key)

    steps = {}
    step_lookups = []
    for s in series:
        lookup = {}
        for i, c in enumerate(s):
            steps[c[0]] = steps.get(c[0], 0) + 1
            lookup[c[0]] = i
        step_lookups.append(lookup)

    steps = [k for k, v in steps.items() if v == len(series)]
    steps = sorted(steps)
    interval = max(1, len(steps) // limit)
    offset = (len(steps) - 1) % interval

    data_series = []
    for si, s in enumerate(series):
        run = get_run(series_inds[si].uuid)
        data = []
        for i in range(offset, len(steps), interval):
            step = steps[i]
            filename = s[step_lookups[si][step]][1]
            data.append((step, _get_numpy_array(run.run_info.artifacts_folder / filename)))
        data_series.append(data)

    return data_series, names
