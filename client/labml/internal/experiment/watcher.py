from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Experiment


class ExperimentWatcher:
    def __init__(self, exp: 'Experiment'):
        self.exp = exp

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.exp.finish('completed')
        elif exc_type == KeyboardInterrupt:
            self.exp.finish('interrupted')
        else:
            self.exp.finish('crashed', str(exc_val))
