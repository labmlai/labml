from typing import Dict

from labml.internal.logger.store.indicators import Indicator


class Writer:
    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        raise NotImplementedError()

    def flush(self):
        pass

    def write_h_parameters(self, hparams: Dict[str, any]):
        pass
