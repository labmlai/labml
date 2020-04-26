import json
import pathlib
from typing import Dict

import numpy as np
import torch.nn

from . import CheckpointSaver, experiment_singleton


class Checkpoint(CheckpointSaver):
    _models: Dict[str, torch.nn.Module]

    def __init__(self, path: pathlib.PurePath):
        self.path = path
        self._models = {}

    def add_models(self, models: Dict[str, torch.nn.Module]):
        """
        ## Set variable for saving and loading
        """
        self._models.update(models)

    def save_model(self,
                   name: str,
                   model: torch.nn.Module,
                   checkpoint_path: pathlib.Path) -> any:
        raise NotImplementedError()

    def save(self, global_step):
        """
        ## Save model as a set of numpy arrays
        """

        checkpoints_path = pathlib.Path(self.path)
        if not checkpoints_path.exists():
            checkpoints_path.mkdir()

        checkpoint_path = checkpoints_path / str(global_step)
        assert not checkpoint_path.exists()

        checkpoint_path.mkdir()

        files = {}
        for name, model in self._models.items():
            files[name] = self.save_model(name, model, checkpoint_path)

        # Save header
        with open(str(checkpoint_path / "info.json"), "w") as f:
            f.write(json.dumps(files))

    def load_model(self,
                   name: str,
                   model: torch.nn.Module,
                   checkpoint_path: pathlib.Path,
                   info: any):
        raise NotImplementedError()

    def load(self, checkpoint_path):
        """
        ## Load model as a set of numpy arrays
        """

        with open(str(checkpoint_path / "info.json"), "r") as f:
            files = json.loads(f.readline())

        # Load each model
        for name, model in self._models.items():
            self.load_model(name, model, checkpoint_path, files[name])

        return True


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
        state = torch.load(str(checkpoint_path / file_name))

        model.load_state_dict(state)


def add_models(models: Dict[str, torch.nn.Module]):
    exp = experiment_singleton()
    if exp.checkpoint_saver is None:
        exp.checkpoint_saver = PyTorchCheckpoint(exp.run.checkpoint_path)

    exp.checkpoint_saver.add_models(models)
