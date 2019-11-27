import json
import pathlib
from typing import Optional, Dict

import numpy as np
import torch.nn

from lab import experiment, util
from lab import logger
from lab.logger_class import CheckpointSaver, sqlite_writer


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

    def __init__(self, *,
                 name: str,
                 python_file: str,
                 comment: str,
                 check_repo_dirty: Optional[bool] = None,
                 is_log_python_file: Optional[bool] = None):
        """
        ### Create the experiment

        :param name: name of the experiment
        :param python_file: `__file__` that invokes this. This is stored in
         the experiments list.
        :param comment: a short description of the experiment
        :param check_repo_dirty: whether not to start the experiment if
         there are uncommitted changes.

        The experiments log keeps track of `python_file`, `name`, `comment` as
         well as the git commit.

        Experiment maintains the locations of checkpoints, logs, etc.
        """

        super().__init__(name=name,
                         python_file=python_file,
                         comment=comment,
                         check_repo_dirty=check_repo_dirty,
                         is_log_python_file=is_log_python_file)

    def _create_checkpoint_saver(self):
        self.__checkpoint_saver = Checkpoint(self.info.checkpoint_path)
        return self.__checkpoint_saver

    def add_models(self, models: Dict[str, torch.nn.Module]):
        """
        ## Set variable for saving and loading
        """
        self.__checkpoint_saver.add_models(models)

    def start_train(self, is_init: bool = True):
        """
        ## Start experiment

        Load a checkpoint or reset based on `global_step`.
        """

        global_step = 0

        if not is_init:
            # load checkpoint if we are starting from middle
            with logger.section("Loading checkpoint"):
                is_successful = self.__checkpoint_saver.load()
                logger.set_successful(is_successful)
                if is_successful:
                    global_step = self.__checkpoint_saver.max_step

        self._start(global_step)

        if global_step == 0:
            # initialize variables and clear summaries if we are starting from scratch
            with logger.section("Clearing summaries"):
                self.clear_summaries()
            with logger.section("Clearing checkpoints"):
                self.clear_checkpoints()

    def start_replay(self):
        """
        ## Start replaying experiment

        Load a checkpoint or reset based on `global_step`.
        """

        with logger.section("Loading checkpoint") as m:
            m.is_successful = self.__checkpoint_saver.load()
