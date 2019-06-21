import numpy as np

from lab import colors


class Writer:
    def write(self, *, global_step: int,
              queues,
              histograms,
              pairs,
              scalars,
              tf_summaries):
        raise NotImplementedError()


class ProgressDictWriter(Writer):
    def __init__(self):
        super().__init__()

        self.indicators = []

    def add_indicator(self, name):
        self.indicators.append(name)

    def write(self, *, global_step: int,
              queues,
              histograms,
              pairs,
              scalars,
              tf_summaries):
        res = dict(global_step=f"{global_step :8,}")

        for k in self.indicators:
            if k in queues:
                if len(queues[k]) == 0:
                    continue
                v = np.mean(queues[k])
            elif k in histograms:
                if len(histograms[k]) == 0:
                    continue
                v = np.mean(histograms[k])
            else:
                if len(scalars[k]) == 0:
                    continue
                v = np.mean(scalars[k])

            res[k] = f"{v :8,.2f}"

        return res


class ScreenWriter(Writer):
    def __init__(self, is_color=True):
        super().__init__()

        self.indicators = []
        self.is_color = is_color

    def add_indicator(self, name):
        self.indicators.append(name)

    def write(self, *, global_step: int,
              queues,
              histograms,
              pairs,
              scalars,
              tf_summaries):
        parts = []

        for k in self.indicators:
            if k in queues:
                if len(queues[k]) == 0:
                    continue
                v = np.mean(queues[k])
            elif k in histograms:
                if len(histograms[k]) == 0:
                    continue
                v = np.mean(histograms[k])
            else:
                if len(scalars[k]) == 0:
                    continue
                v = np.mean(scalars[k])

            parts.append((f" {k}: ", None))
            if self.is_color:
                parts.append((f"{v :8,.2f}", colors.Style.bold))
            else:
                parts.append((f"{v :8,.2f}", None))

        return parts