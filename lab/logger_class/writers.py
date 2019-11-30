import numpy as np

from lab import colors


class Writer:
    def add_indicator(self, name: str, *,
                      indicator_type: str,
                      queue_limit: int,
                      is_print: bool):
        pass

    def write(self, *, global_step: int,
              queues,
              histograms,
              pairs,
              scalars):
        raise NotImplementedError()


class ScreenWriter(Writer):
    def __init__(self, is_color=True):
        super().__init__()

        self.indicators = []
        self.is_color = is_color

    def add_indicator(self, name: str, *,
                      indicator_type: str,
                      queue_limit: int,
                      is_print: bool):
        if is_print:
            self.indicators.append(name)

    def write(self, *, global_step: int,
              queues,
              histograms,
              pairs,
              scalars):
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
