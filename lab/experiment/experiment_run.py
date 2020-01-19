import time
from pathlib import Path, PurePath
from typing import List, Dict, Optional, Set

import numpy as np

from .. import util, logger
from ..logger.colors import Text


def _struct_time_to_time(t: time.struct_time):
    return f"{t.tm_hour :02}:{t.tm_min :02}:{t.tm_sec :02}"


def _struct_time_to_date(t: time.struct_time):
    return f"{t.tm_year :04}-{t.tm_mon :02}-{t.tm_mday :02}"


_GLOBAL_STEP = 'global_step'


class RunInfo:
    def __init__(self, *,
                 index: int,
                 python_file: str,
                 trial_date: str,
                 trial_time: str,
                 comment: str,
                 commit: Optional[str] = None,
                 commit_message: Optional[str] = None,
                 is_dirty: bool = True,
                 experiment_path: PurePath,
                 start_step: int = 0):
        self.index = index
        self.commit = commit
        self.is_dirty = is_dirty
        self.python_file = python_file
        self.trial_date = trial_date
        self.trial_time = trial_time
        self.comment = comment
        self.commit_message = commit_message
        self.start_step = start_step

        self.experiment_path = experiment_path
        self.run_path = experiment_path / str(index)
        self.checkpoint_path = self.run_path / "checkpoints"
        self.numpy_path = self.run_path / "numpy"

        self.diff_path = self.run_path / "source.diff"

        self.sqlite_path = self.run_path / "sqlite.db"
        self.tensorboard_log_path = self.run_path / "tensorboard"

        self.info_path = self.run_path / "run.yaml"
        self.indicators_path = self.run_path / "indicators.yaml"
        self.configs_path = self.run_path / "configs.yaml"

    @classmethod
    def from_dict(cls, experiment_path: PurePath, data: Dict[str, any]):
        """
        ## Create a new trial from a dictionary
        """
        params = dict(experiment_path=experiment_path)
        params.update(data)
        return cls(**params)

    def to_dict(self):
        """
        ## Convert trial to a dictionary for saving
        """
        return dict(
            index=self.index,
            python_file=self.python_file,
            trial_date=self.trial_date,
            trial_time=self.trial_time,
            comment=self.comment,
            commit=self.commit,
            commit_message=self.commit_message,
            is_dirty=self.is_dirty,
            start_step=self.start_step
        )

    def pretty_print(self) -> List[str]:
        """
        ## 🎨 Pretty print trial for the python file header
        """

        # Trial information
        commit_status = "[dirty]" if self.is_dirty else "[clean]"
        res = [
            f"{self.trial_date} {self.trial_time}",
            self.comment,
            f"[{commit_status}]: {self.commit_message}",
            f"start_step: {self.start_step}"
        ]

        return res

    def __str__(self):
        return f"{self.__class__.__name__}(comment=\"{self.comment}\"," \
               f" commit=\"{self.commit_message}\"," \
               f" date={self.trial_date}, time={self.trial_time})"

    def __repr__(self):
        return self.__str__()


class Run(RunInfo):
    """
    # Trial 🏃‍

    Every trial in an experiment has same configs.
    It's just multiple runs.

    A new trial will replace checkpoints and TensorBoard summaries
    or previous trials, you should make a copy if needed.
    The performance log in `trials.yaml` is not replaced.

    You should run new trials after bug fixes or to see performance is
     consistent.

    If you want to try different configs, create multiple experiments.
    """

    diff: Optional[str]

    def __init__(self, *,
                 index: int,
                 python_file: str,
                 trial_date: str,
                 trial_time: str,
                 comment: str,
                 commit: Optional[str] = None,
                 commit_message: Optional[str] = None,
                 is_dirty: bool = True,
                 experiment_path: PurePath,
                 start_step: int = 0):
        super().__init__(python_file=python_file, trial_date=trial_date, trial_time=trial_time,
                         comment=comment, index=index, experiment_path=experiment_path,
                         commit=commit, commit_message=commit_message, is_dirty=is_dirty,
                         start_step=start_step)

    @classmethod
    def create(cls, *,
               experiment_path: PurePath,
               python_file: str,
               trial_time: time.struct_time,
               comment: str):
        """
        ## Create a new trial
        """
        runs = [int(child.name) for child in Path(experiment_path).iterdir()]
        runs.sort()

        if len(runs) > 0:
            this_run = runs[-1] + 1
        else:
            this_run = 1

        return cls(python_file=python_file,
                   trial_date=_struct_time_to_date(trial_time),
                   trial_time=_struct_time_to_time(trial_time),
                   index=this_run,
                   experiment_path=experiment_path,
                   comment=comment)

    def save_info(self):
        run_path = Path(self.run_path)
        if not run_path.exists():
            run_path.mkdir(parents=True)

        with open(str(self.info_path), "w") as file:
            file.write(util.yaml_dump(self.to_dict()))

        if self.diff is not None:
            with open(str(self.diff_path), "w") as f:
                f.write(self.diff)


def get_checkpoints(experiment_path: PurePath, run_index: int):
    run_path = experiment_path / str(run_index)
    checkpoint_path = Path(run_path / "checkpoints")
    if not checkpoint_path.exists():
        return {}

    return {int(child.name) for child in Path(checkpoint_path).iterdir()}


def get_runs(experiment_path: PurePath):
    return {int(child.name) for child in Path(experiment_path).iterdir()}


def get_run_checkpoint(experiment_path: PurePath,
                       run_index: int = -1, checkpoint: int = -1,
                       skip_index: Set[int] = None):
    if skip_index is None:
        skip_index = {}
    runs = get_runs(experiment_path)
    runs.difference_update(skip_index)

    if len(runs) == 0:
        return None, None, None

    if run_index < 0:
        required_ri = np.max(list(runs)) + run_index + 1
    else:
        required_ri = run_index

    for ri in range(required_ri, -1, -1):
        if ri not in runs:
            continue

        checkpoints = get_checkpoints(experiment_path, ri)
        if len(checkpoints) == 0:
            continue

        if checkpoint < 0:
            required_ci = np.max(list(checkpoints)) + checkpoint + 1
        else:
            required_ci = checkpoint

        for ci in range(required_ci, -1, -1):
            if ci not in checkpoints:
                continue

            return required_ri, ri, ci

    return required_ri, None, None


def get_last_run_checkpoint(experiment_path: PurePath,
                            run_index: int = -1,
                            checkpoint: int = -1,
                            skip_index: Set[int] = None):
    required_ri, run_index, checkpoint = get_run_checkpoint(experiment_path, run_index, checkpoint, skip_index)

    if run_index is None:
        logger.log("Couldn't find a previous run/checkpoint")
        return None, None

    if required_ri > run_index:
        logger.log(f"Skipped runs [{run_index + 1}...{required_ri}]", Text.warning)

    logger.log(["Selected ",
                ("run", Text.key),
                " = ",
                (run_index, Text.value),
                " ",
                ("checkpoint", Text.key),
                " = ",
                (checkpoint, Text.value)])

    run_path = experiment_path / str(run_index)
    checkpoint_path = run_path / "checkpoints"
    return checkpoint_path / str(checkpoint), checkpoint
