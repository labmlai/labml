from argparse import Namespace
from typing import Any, Dict, Optional, Union

import torch
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

from labml import experiment, tracker, lab
from labml.internal.experiment import experiment_singleton


class LabMLLightningLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def experiment(self):
        return None

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        experiment.configs(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None) -> None:
        if step is None:
            tracker.add_global_step()
            tracker.save(metrics)
        else:
            tracker.save(step, metrics)

    def reset_experiment(self):
        pass

    @property
    def save_dir(self) -> Optional[str]:
        return str(lab.get_experiments_path())

    @property
    def name(self) -> str:
        return str(experiment_singleton().run.name)

    @property
    def version(self) -> str:
        return experiment_singleton().run.uuid
