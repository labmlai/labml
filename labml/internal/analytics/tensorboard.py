from typing import List

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.distribution import compressor

from .analytics import Analytics, BASIS_POINTS, Event


class TensorBoardAnalytics(Analytics):
    def __init__(self, log_path):
        self.event_acc = EventAccumulator(str(log_path), size_guidance={'tensors': 1000})

    def load(self):
        self.event_acc.Reload()

    def tensor(self, name) -> List[Event]:
        name = name.replace('.', '/')
        events = self.event_acc.Tensors(name)
        return [Event(e.step, tf.make_ndarray(e.tensor_proto)) for e in events]

    def summarize(self, events: List[Event]):
        step = np.mean([e.step for e in events])
        values = np.sort([e.tensor for e in events])
        basis_points = np.percentile(values, BASIS_POINTS)

        return np.concatenate(([step], basis_points))

    def summarize_compressed_histogram(self, events: List[Event]):
        basis_points = [int(b) for b in np.multiply(BASIS_POINTS, 100)]
        results = []
        for e in events:
            buckets = compressor.compress_histogram(e.tensor)
            assert (len(buckets) == len(basis_points))
            for i, c in enumerate(buckets):
                assert (c.basis_point == basis_points[i])
            results.append([e.step] + [c.value for c in buckets])

        return np.asarray(results)
