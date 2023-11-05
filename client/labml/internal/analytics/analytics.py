from typing import List, Optional, NamedTuple

import numpy as np

BASIS_POINTS = [
    0,
    6.68,
    15.87,
    30.85,
    50.00,
    69.15,
    84.13,
    93.32,
    100.00
]


class Event(NamedTuple):
    step: int
    tensor: np.ndarray


class Analytics:
    def summarize(self, events):
        step = np.mean([e[0] for e in events])
        values = np.sort([e[1] for e in events])
        basis_points = np.percentile(values, BASIS_POINTS)

        return np.concatenate(([step], basis_points))

    def summarize_scalars(self, events: List[any], points: Optional[int] = 100):
        # Shrink to 100 histograms
        if points is None:
            interval = 1
        else:
            interval = max(1, len(events) // points)

        merged = []
        results = []
        for i, e in enumerate(events):
            if i > 0 and (i + 1) % interval == 0:
                results.append(self.summarize(merged))
                merged = []
            merged.append(e)
        if len(merged) > 0:
            results.append(self.summarize(merged))

        return np.array(results)
