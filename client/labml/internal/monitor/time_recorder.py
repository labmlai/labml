import time
from typing import NamedTuple

import numpy as np


class Timer:
    def __init__(self, tr: 'TimeRecorder', name: str, idx: int):
        self.tr = tr
        self.idx = idx
        self.name = name
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def done(self):
        self.tr.set_time(self.name, self.idx, time.time() - self.start_time)


class Summary(NamedTuple):
    events: int
    mean: float
    std: float
    min: float
    max: float
    ongoing: int


class TimeRecorder:
    def __init__(self):
        self.times = {}

    def record_time(self, name: str):
        if name not in self.times:
            self.times[name] = {}

        timer = Timer(self, name, len(self.times[name]))
        self.times[name][timer.idx] = timer
        timer.start()

        return timer

    def set_time(self, name, idx, td):
        self.times[name][idx] = td

    def get_summary(self, name: str, ignore_first: int, ignore_last: int):
        times = self.times[name]
        times = [(int(idx), v) for idx, v in times.items()]
        times.sort()
        times = [td for _, td in times]
        if ignore_first:
            times = times[ignore_first:]
        if ignore_last:
            times = times[:-ignore_last]

        time_values = [td for td in times if isinstance(td, float)]
        ongoing = [td for td in times if isinstance(td, Timer)]

        time_values = np.array(time_values)

        if len(time_values) == 0:
            return Summary(
                events=len(time_values),
                mean=0,
                std=0,
                min=0,
                max=0,
                ongoing=len(ongoing),
            )

        return Summary(
            events=len(time_values),
            mean=time_values.mean(),
            std=time_values.std(),
            min=time_values.min(),
            max=time_values.max(),
            ongoing=len(ongoing),
        )

    def get_times(self, ignore_first: int, ignore_last: int):
        return {k: self.get_summary(k, ignore_first, ignore_last) for k in self.times}
