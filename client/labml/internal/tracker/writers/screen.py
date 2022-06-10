import math
from typing import Dict

import numpy as np

from labml import logger
from labml.internal.util.values import to_numpy
from labml.logger import Text
from .. import Writer, Indicator
from ..indicators import artifacts
from ..indicators import numeric


class ScreenWriter(Writer):
    def __init__(self):
        super().__init__()

        self._estimates = {}
        self._beta = 0.9
        self._beta_pow = {}
        self._last_printed_value = {}

    def update_estimate(self, k, v):
        if k not in self._estimates or not np.isfinite(self._estimates[k]):
            self._estimates[k] = 0
            self._beta_pow[k] = 1.

        self._estimates[k] *= self._beta
        self._estimates[k] += (1 - self._beta) * v
        self._beta_pow[k] *= self._beta

    def get_empty_string(self, length, decimals):
        return ' ' * (length - 2 - decimals) + '-.' + '-' * decimals

    def get_value_string(self, k, v):
        if k not in self._estimates:
            assert v is None
            return self.get_empty_string(8, 2)

        estimate = self._estimates[k] / (1 - self._beta_pow[k])
        if abs(estimate) < 1e-9 or not np.isfinite(estimate):
            lg = 0
        else:
            lg = int(np.ceil(np.log10(abs(estimate)))) + 1

        if lg >= 7:
            fmt = "{v:9.3e}"
        else:
            decimals = np.clip(7 - lg, 1, 6)
            fmt = "{v:8,." + str(decimals) + "f}"

        if v is None:
            return self.get_empty_string(8, decimals)
        else:
            return fmt.format(v=v)

    @staticmethod
    def __format_artifact(length: int, value: str):
        fmt = "{v:>" + str(length + 1) + "}"
        return fmt.format(v=value)

    def _get_indicator_string(self, indicators: Dict[str, Indicator]):
        parts = []

        for ind in indicators.values():
            if not isinstance(ind, numeric.NumericIndicator):
                continue
            if not ind.is_print:
                continue

            parts.append((f" {ind.name}: ", None))

            if not ind.is_empty():
                v = ind.get_mean()
                self.update_estimate(ind.name, v)
                value = self.get_value_string(ind.name, v)
                self._last_printed_value[ind.name] = value
                parts.append((value, Text.value))
            elif ind.name in self._last_printed_value:
                value = self._last_printed_value[ind.name]
                parts.append((value, Text.subtle))
            else:
                value = self.get_value_string(ind.name, None)
                parts.append((value, Text.subtle))

        return parts

    def _print_artifacts_list(self, table: Dict[str, int], artifacts: Dict[str, artifacts.Artifact]):
        order = list(table.keys())
        if not len(order):
            return

        keys = {k for name in order for k in artifacts[name].keys()}
        for k in keys:
            for name in order:
                value = self._get_artifact_string(artifacts[name], k)
                logger.log([(name, Text.key),
                            ": ",
                            (value, Text.value)])

    def _print_artifacts_table(self, table: Dict[str, int], artifacts: Dict[str, artifacts.Artifact]):
        order = list(table.keys())
        if not len(order):
            return

        keys = []
        keys_set = set()

        for name in order:
            for k in artifacts[name].keys():
                if k not in keys_set:
                    keys_set.add(k)
                    keys.append(k)

        parts = [self.__format_artifact(table[name], name) for name in order]
        logger.log('|'.join(parts), Text.heading)

        for k in keys:
            parts = []
            for name in order:
                value = self._get_artifact_string(artifacts[name], k)
                parts.append(self.__format_artifact(table[name], value))
            logger.log('|'.join(parts), Text.value)

    def _print_artifact(self, indicator: artifacts.Artifact):
        if isinstance(indicator, artifacts.Image):
            try:
                import matplotlib.pyplot as plt
            except (ImportError, ModuleNotFoundError):
                plt = None

            if plt is None:
                logger.log(('matplotlib', logger.Text.highlight),
                           ' not found. So cannot display images')
            images = indicator.get_images()
            n_images = len(images)
            cols = max(1, int(math.sqrt(n_images)))
            fig: plt.Figure
            fig, axs = plt.subplots((n_images + cols - 1) // cols, cols,
                                    sharex='all', sharey='all',
                                    figsize=(8, 10))
            from labml import tracker
            fig.suptitle(f'{indicator.name}-{tracker.get_global_step()}')
            for i, img in enumerate(images):
                if len(images) > 1:
                    ax: plt.Axes = axs[i // cols, i % cols]
                else:
                    ax = axs
                if img.shape[0] == 1:
                    img = img[0, :, :]
                else:
                    img = img.transpose(1, 2, 0)

                ax.imshow(img)
            plt.show()
        elif isinstance(indicator, artifacts.Text):
            logger.log(indicator.name, Text.heading)
            for t in indicator.get_values().values():
                logger.log(t, Text.value)

    def _get_artifact_print_length(self, indicator: artifacts.Artifact):
        if isinstance(indicator, artifacts.IndexedText):
            return max((len(v) for v in indicator.get_values().values()))
        else:
            return None

    def _get_artifact_string(self, indicator: artifacts.Artifact, key: str):
        if isinstance(indicator, artifacts.IndexedText):
            return indicator.get_value(key)
        else:
            return None

    def _print_artifacts(self, indicators: Dict[str, Indicator]):
        table = {}
        artifact_inds = {}
        for ind in indicators.values():
            if not isinstance(ind, artifacts.Artifact):
                continue
            if not ind.is_print:
                continue
            if ind.is_empty():
                continue
            if not ind.is_indexed:
                self._print_artifact(ind)
                continue

            table[ind.name] = self._get_artifact_print_length(ind)
            artifact_inds[ind.name] = ind

        if sum(table.values()) > 100:
            self._print_artifacts_list(table, artifact_inds)
        else:
            self._print_artifacts_table(table, artifact_inds)

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):

        self._print_artifacts(indicators)

        return self._get_indicator_string(indicators)
