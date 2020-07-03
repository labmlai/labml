import pathlib
from typing import Dict

import numpy as np
import torch.nn

from . import ModelSaver, experiment_singleton


class NumpyModelSaver(ModelSaver):
    """
    Deprecated: left for backward compatibility only
    Remove around August 2020
    """

    def __init__(self, name: str, model: torch.nn.Module):
        self.name = name
        self.model = model

    def save(self, checkpoint_path: pathlib.Path) -> any:
        state: Dict[str, torch.Tensor] = self.model.state_dict()
        files = {}
        for key, tensor in state.items():
            if key == "_metadata":
                continue

            file_name = f"{self.name}_{key}.npy"
            files[key] = file_name

            np.save(str(checkpoint_path / file_name), tensor.cpu().numpy())

        return files

    def load(self, checkpoint_path: pathlib.Path, info: any):
        state: Dict[str, torch.Tensor] = self.model.state_dict()
        for key, tensor in state.items():
            file_name = info[key]
            saved = np.load(str(checkpoint_path / file_name))
            saved = torch.from_numpy(saved).to(tensor.device)
            state[key] = saved

        self.model.load_state_dict(state)


class PyTorchModelSaver(ModelSaver):
    def __init__(self, name: str, model: torch.nn.Module):
        self.name = name
        self.model = model

    def save(self, checkpoint_path: pathlib.Path) -> any:
        state = self.model.state_dict()
        file_name = f"{self.name}.pth"
        torch.save(state, str(checkpoint_path / file_name))
        return file_name

    def load(self, checkpoint_path: pathlib.Path, info: any):
        file_name: str = info
        try:
            sample_param = next(self.model.parameters())
            device = sample_param.device
        except StopIteration:
            device = torch.device('cpu')

        state = torch.load(str(checkpoint_path / file_name), map_location=device)

        self.model.load_state_dict(state)


def add_models(models: Dict[str, torch.nn.Module]):
    exp = experiment_singleton()
    savers = {name: PyTorchModelSaver(name, model) for name, model in models.items()}

    exp.checkpoint_saver.add_savers(savers)
