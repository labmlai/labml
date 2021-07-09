import dataclasses
from typing import List

import torch

from labml import tracker
from . import Metric


@dataclasses.dataclass
class CollectorState:
    data: List

    def reset(self):
        self.data = []


class Collector(Metric):
    data: CollectorState

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __call__(self, data: torch.Tensor):
        self.data.data.append(data)

    def create_state(self):
        return CollectorState([])

    def set_state(self, data: any):
        self.data = data

    def on_epoch_start(self):
        self.data.reset()

    def on_epoch_end(self):
        if not self.data.data:
            return
        tracker.add(f"{self.name}.", torch.cat(self.data.data, dim=0))
