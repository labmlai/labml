import dataclasses

import torch

from labml import tracker
from . import Metric


@dataclasses.dataclass
class RecallPrecisionState:
    fn: int = 0
    fp: int = 0
    tn: int = 0
    tp: int = 0

    def reset(self):
        self.fn = 0
        self.fp = 0
        self.tn = 0
        self.tp = 0

    @property
    def total(self):
        return self.fn + self.fp + self.tn + self.tp


class RecallPrecision(Metric):
    data: RecallPrecisionState

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        pred = output.view(-1) > 0
        target = target.view(-1)

        self.data.fn += ((pred == 0) & (target == 1)).sum().item()
        self.data.fp += ((pred == 1) & (target == 0)).sum().item()
        self.data.tn += ((pred == 0) & (target == 0)).sum().item()
        self.data.tp += ((pred == 1) & (target == 1)).sum().item()

    def create_state(self):
        return RecallPrecisionState()

    def set_state(self, data: any):
        self.data = data

    def on_epoch_start(self):
        self.data.reset()

    def on_epoch_end(self):
        self.track()

    def track(self):
        if self.data.tp + self.data.fp > 0:
            tracker.add("prcn.", self.data.tp / (self.data.tp + self.data.fp))
        if self.data.tp + self.data.fn > 0:
            tracker.add("recl.", self.data.tp / (self.data.tp + self.data.fn))
