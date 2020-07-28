from typing import Dict

import numpy as np

from labml import logger
from labml.internal.logger import Writer, Indicator
from labml.internal.logger.store import Artifact
from labml.internal.logger.store.indicators.numeric import NumericIndicator
from labml.logger import Text


class ScreenWriter(Writer):
    def __init__(self):
        super().__init__()

        self._estimates = {}
        self._beta = 0.9
        self._beta_pow = {}
        self._last_printed_value = {}

    def update_estimate(self, k, v):
        if k not in self._estimates:
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
        if abs(estimate) < 1e-9 or np.isnan(estimate):
            lg = 0
        else:
            lg = int(np.ceil(np.log10(abs(estimate)))) + 1

        decimals = 7 - lg
        decimals = max(1, decimals)
        decimals = min(6, decimals)

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
            if not isinstance(ind, NumericIndicator):
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

    @staticmethod
    def _print_artifacts_list(table: Dict[str, int], artifacts: Dict[str, Artifact]):
        order = list(table.keys())
        if not len(order):
            return

        keys = {k for name in order for k in artifacts[name].keys()}
        for k in keys:
            for name in order:
                value = artifacts[name].get_string(k, artifacts)
                logger.log([(name, Text.key),
                            ": ",
                            (value, Text.value)])

    def _print_artifacts_table(self, table: Dict[str, int], artifacts: Dict[str, Artifact]):
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
                value = artifacts[name].get_string(k, artifacts)
                parts.append(self.__format_artifact(table[name], value))
            logger.log('|'.join(parts), Text.value)

    def _print_artifacts(self, indicators: Dict[str, Indicator]):
        table  = {}
        artifacts = {}
        for ind in indicators.values():
            if not isinstance(ind, Artifact):
                return
            if not ind.is_print:
                continue
            if ind.is_empty():
                continue
            if not ind.is_indexed:
                ind.print_all()
                continue

            table[ind.name] = ind.get_print_length()
            artifacts[ind.name] = ind

        if sum(table.values()) > 100:
            self._print_artifacts_list(table, artifacts)
        else:
            self._print_artifacts_table(table, artifacts)

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):

        self._print_artifacts(indicators)

        return self._get_indicator_string(indicators)