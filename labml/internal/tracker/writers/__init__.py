from typing import Dict

from ..indicators import Indicator


class Writer:
    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        raise NotImplementedError()

    def flush(self):
        pass

    def write_h_parameters(self, hparams: Dict[str, any]):
        pass

    def save_indicators(self, dot_indicators: Dict[str, Indicator], indicators: Dict[str, Indicator]):
        pass
