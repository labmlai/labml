from typing import Optional, Set, Dict, List, Union

import numpy as np
import torch

from lab.internal.experiment import \
    create_experiment as _create_experiment, \
    experiment_singleton as _experiment_singleton
from lab.internal.experiment.pytorch import add_models as _add_models
from lab.configs import BaseConfigs


def save_checkpoint():
    _experiment_singleton().save_checkpoint()


def create(*,
           name: Optional[str] = None,
           python_file: Optional[str] = None,
           comment: Optional[str] = None,
           writers: Set[str] = None,
           ignore_callers: Set[str] = None,
           tags: Optional[Set[str]] = None):
    _create_experiment(name=name,
                       python_file=python_file,
                       comment=comment,
                       writers=writers,
                       ignore_callers=ignore_callers,
                       tags=tags)


def add_pytorch_models(models: Dict[str, torch.nn.Module]):
    _add_models(models)


def calculate_configs(
        configs: Optional[BaseConfigs],
        configs_dict: Dict[str, any] = None,
        run_order: Optional[List[Union[List[str], str]]] = None):
    _experiment_singleton().calc_configs(configs, configs_dict, run_order)


def start():
    _experiment_singleton().start()


def load(run_uuid: str,
         checkpoint: Optional[int] = None):
    _experiment_singleton().start(run_uuid=run_uuid, checkpoint=checkpoint)


def save_numpy(name: str, array: np.ndarray):
    """
    Save a single numpy array.
    This is used to save processed data
    """
    numpy_path = _experiment_singleton().run.numpy_path

    if not numpy_path.exists():
        numpy_path.mkdir(parents=True)
    file_name = name + ".npy"
    np.save(str(numpy_path / file_name), array)
