import json
import pathlib
from typing import Optional, Dict, Set

import numpy as np
import torch.nn

from lab._internal import experiment
from lab._internal.logger.internal import CheckpointSaver


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


class Experiment(experiment.Experiment):
    r"""
    Concrete implementation of :class:`lab.experiment.Experiment` for PyTorch
    experiments
    """

    __checkpoint_saver: Checkpoint

    def __init__(self, *,
                 name: Optional[str] = None,
                 python_file: Optional[str] = None,
                 comment: Optional[str] = None,
                 writers: Set[str] = None,
                 ignore_callers: Set[str] = None,
                 tags: Optional[Set[str]] = None):

        super().__init__(name=name,
                         python_file=python_file,
                         comment=comment,
                         writers=writers,
                         ignore_callers=ignore_callers,
                         tags=tags)

    def _create_checkpoint_saver(self):
        self.__checkpoint_saver = PyTorchCheckpoint(self.run.checkpoint_path)
        return self.__checkpoint_saver

    def add_models(self, models: Dict[str, torch.nn.Module]):
        """
        Set variables for saving and loading

        Arguments:
            models (Dict[str, torch.nn.Module]): a dictionary of torch modules
                used in the experiment. These will be saved with :func:`lab.logger.save_checkpoint`
                and loaded with :meth:`lab.experiment.Experiment.start`.

        """

        self.__checkpoint_saver.add_models(models)

    def _load_checkpoint(self, checkpoint_path: pathlib.PurePath):
        self.__checkpoint_saver.load(checkpoint_path)
