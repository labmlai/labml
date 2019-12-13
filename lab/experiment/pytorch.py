import json
import pathlib
from typing import Optional, Dict, Set

import numpy as np
import torch.nn

from lab import experiment, util
from lab.configs import Configs
from lab.logger.internal import CheckpointSaver


class Checkpoint(CheckpointSaver):
    max_step: Optional[int]
    __models: Dict[str, torch.nn.Module]

    def __init__(self, path: pathlib.PurePath):
        self.path = path
        self.max_step = None
        self.__models = {}

    def add_models(self, models: Dict[str, torch.nn.Module]):
        """
        ## Set variable for saving and loading
        """
        self.__models.update(models)

    def save(self, global_step, args):
        self._save(global_step)

    def _save(self, global_step):
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
        for name, model in self.__models.items():
            state: Dict[str, torch.Tensor] = model.state_dict()
            files[name] = {}
            for key, tensor in state.items():
                if key == "_metadata":
                    continue

                file_name = f"{name}_{key}.npy"
                files[name][key] = file_name

                np.save(str(checkpoint_path / file_name), tensor.cpu().numpy())

        # Save header
        with open(str(checkpoint_path / "info.json"), "w") as f:
            f.write(json.dumps(files))

        # Delete old checkpoints
        for c in checkpoints_path.iterdir():
            if c.name != checkpoint_path.name:
                util.rm_tree(c)

    def load(self):
        """
        ## Load model as a set of numpy arrays
        """

        checkpoints_path = pathlib.Path(self.path)
        if not checkpoints_path.exists():
            return False

        max_step = -1
        for c in checkpoints_path.iterdir():
            max_step = max(max_step, int(c.name))

        if max_step < 0:
            return False

        checkpoint_path = checkpoints_path / str(max_step)

        with open(str(checkpoint_path / "info.json"), "r") as f:
            files = json.loads(f.readline())

        # Load each variable
        for name, model in self.__models.items():
            state: Dict[str, torch.Tensor] = model.state_dict()
            for key, tensor in state.items():
                file_name = files[name][key]
                saved = np.load(str(checkpoint_path / file_name))
                saved = torch.from_numpy(saved).to(tensor.device)
                state[key] = saved

            model.load_state_dict(state)

        self.max_step = max_step
        return True


class Experiment(experiment.Experiment):
    """
    ## Experiment

    Each experiment has different configurations or algorithms.
    An experiment can have multiple trials.
    """

    __checkpoint_saver: Checkpoint

    def __init__(self, *,
                 name: Optional[str] = None,
                 python_file: Optional[str] = None,
                 comment: Optional[str] = None,
                 configs: Optional[Configs] = None,
                 writers: Set[str] = None):
        """
        ### Create the experiment

        :param name: name of the experiment
        :param python_file: `__file__` that invokes this. This is stored in
         the experiments list.
        :param comment: a short description of the experiment

        The experiments log keeps track of `python_file`, `name`, `comment` as
         well as the git commit.

        Experiment maintains the locations of checkpoints, logs, etc.
        """

        super().__init__(name=name,
                         python_file=python_file,
                         comment=comment,
                         configs=configs,
                         writers=writers)

    def _create_checkpoint_saver(self):
        self.__checkpoint_saver = Checkpoint(self.run.checkpoint_path)
        return self.__checkpoint_saver

    def add_models(self, models: Dict[str, torch.nn.Module]):
        """
        ## Set variable for saving and loading
        """
        self.__checkpoint_saver.add_models(models)

    def _load_checkpoint(self):
        is_successful = self.__checkpoint_saver.load()
        if not is_successful:
            return None
        return self.__checkpoint_saver.max_step
