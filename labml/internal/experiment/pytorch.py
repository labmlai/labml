import pathlib
from typing import Dict

import numpy as np
import torch.nn

from . import Checkpoint, experiment_singleton


class NumpyCheckpoint(Checkpoint):
    """
    Deprecated: left for backward compatibility only
    Remove around August 2020
    """

    def save_model(self,
                   name: str,
                   model: torch.nn.Module,
                   checkpoint_path: pathlib.Path) -> any:
        state: Dict[str, torch.Tensor] = model.state_dict()
        files = {}
        for key, tensor in state.items():
            if key == "_metadata":
                continue

            file_name = f"{name}_{key}.npy"
            files[key] = file_name

            np.save(str(checkpoint_path / file_name), tensor.cpu().numpy())

        return files

    def load_model(self,
                   name: str,
                   model: torch.nn.Module,
                   checkpoint_path: pathlib.Path,
                   info: any):
        state: Dict[str, torch.Tensor] = model.state_dict()
        for key, tensor in state.items():
            file_name = info[key]
            saved = np.load(str(checkpoint_path / file_name))
            saved = torch.from_numpy(saved).to(tensor.device)
            state[key] = saved

        model.load_state_dict(state)


class PyTorchCheckpoint(Checkpoint):
    def save_model(self,
                   name: str,
                   model: torch.nn.Module,
                   checkpoint_path: pathlib.Path) -> any:
        state = model.state_dict()
        file_name = f"{name}.pth"
        torch.save(state, str(checkpoint_path / file_name))
        return file_name

    def load_model(self,
                   name: str,
                   model: torch.nn.Module,
                   checkpoint_path: pathlib.Path,
                   info: any):
        file_name: str = info
        try:
            sample_param = next(model.parameters())
            device = sample_param.device
        except StopIteration:
            device = torch.device('cpu')

        state = torch.load(str(checkpoint_path / file_name), map_location=device)

        model.load_state_dict(state)


def add_models(models: Dict[str, torch.nn.Module]):
    exp = experiment_singleton()
    if exp.checkpoint_saver is None:
        exp.checkpoint_saver = PyTorchCheckpoint(exp.run.checkpoint_path)

    exp.checkpoint_saver.add_models(models)
