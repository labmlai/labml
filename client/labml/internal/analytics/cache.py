from pathlib import PurePath
from typing import List, Tuple

import numpy as np

from labml.internal.analytics.indicators import Indicator, Run, RunsSet

_RUNS = {}

_NUMPY_ARRAYS = {}


def get_run(uuid: str) -> Run:
    if uuid not in _RUNS:
        _RUNS[uuid] = Run(uuid)

    return _RUNS[uuid]


def get_experiment_runs(experiment_name: str) -> List[str]:
    return RunsSet().get_runs(experiment_name)


def get_name(ind: Indicator) -> List[str]:
    run = get_run(ind.uuid)
    return [run.name, ind.uuid[-5:], run.run_info.comment, ind.key]


def _get_numpy_array(path: PurePath):
    path = str(path)
    if path not in _NUMPY_ARRAYS:
        _NUMPY_ARRAYS[path] = np.load(path)

    return _NUMPY_ARRAYS[path]


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
