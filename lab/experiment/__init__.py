import pathlib
import time
from typing import Dict, Optional

import git
import numpy as np

from lab import colors, util
from lab import logger
from lab.commenter import Commenter
from lab.experiment.experiment_trial import Trial
from lab.lab import Lab

commenter = Commenter(
    comment_start='"""',
    comment_end='"""',
    add_start='```trial',
    add_end='```'
)


class ExperimentInfo:
    """
    ## Experiment Information

    This class keeps track of paths.
    """

    def __init__(self, lab: Lab, name: str):
        """
        ### Initialize
        """
        self.name = name
        self.experiment_path = lab.experiments / name
        self.checkpoint_path = self.experiment_path / "checkpoints"
        self.npy_path = self.experiment_path / "npy"
        self.model_file = self.checkpoint_path / 'model'

        self.diff_path = self.experiment_path / "diffs"

        self.summary_path = self.experiment_path / "log"
        self.sqlite_path = self.experiment_path / "data.sqlite"
        self.screenshots_path = self.experiment_path / 'screenshots'
        self.trials_log_file = self.experiment_path / "trials.yaml"

    def exists(self) -> bool:
        """
        ### Check is this experiment results exists
        """
        p = pathlib.Path(self.summary_path)
        return p.exists()


class Experiment:
    """
    ## Experiment

    Each experiment has different configurations or algorithms.
    An experiment can have multiple trials.
    """

    def __init__(self, *,
                 name: str,
                 python_file: str,
                 comment: str,
                 check_repo_dirty: Optional[bool],
                 is_log_python_file: Optional[bool]):
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

        self.lab = Lab(python_file)

        if check_repo_dirty is None:
            check_repo_dirty = self.lab.check_repo_dirty
        if is_log_python_file is None:
            is_log_python_file = self.lab.is_log_python_file

        self.info = ExperimentInfo(self.lab, name)

        self.check_repo_dirty = check_repo_dirty

        experiment_path = pathlib.Path(self.info.experiment_path)
        if not experiment_path.exists():
            experiment_path.mkdir(parents=True)

        self.trial = Trial.new_trial(
            python_file=python_file,
            trial_time=time.localtime(),
            comment=comment)

        repo = git.Repo(self.lab.path)

        self.trial.commit = repo.head.commit.hexsha
        self.trial.commit_message = repo.head.commit.message.strip()
        self.trial.is_dirty = repo.is_dirty()
        self.trial.diff = repo.git.diff()

        checkpoint_saver = self._create_checkpoint_saver()
        logger.set_checkpoint_saver(checkpoint_saver)

    def _create_checkpoint_saver(self):
        return None

    def print_info_and_check_repo(self):
        """
        ## 🖨 Print the experiment info and check git repo status
        """
        logger.log_color([
            (self.info.name, colors.Style.bold)
        ])
        logger.log_color([
            ("\t", None),
            (self.trial.comment, colors.BrightColor.cyan)
        ])
        logger.log_color([
            ("\t", None),
            ("[dirty]" if self.trial.is_dirty else "[clean]", None),
            (": ", None),
            (f"\"{self.trial.commit_message.strip()}\"", colors.BrightColor.orange)
        ])

        # Exit if git repository is dirty
        if self.check_repo_dirty and self.trial.is_dirty:
            logger.log("Cannot trial an experiment with uncommitted changes. ",
                            new_line=False)
            logger.log("[FAIL]", color=colors.BrightColor.red)
            exit(1)

    def save_npy(self, array: np.ndarray, name: str):
        """
        ## Save a single numpy array

        This is used to save processed data
        """
        npy_path = pathlib.Path(self.info.npy_path)
        npy_path.mkdir(parents=True)
        file_name = name + ".npy"
        np.save(str(self.info.npy_path / file_name), array)

    def load_npy(self, name: str):
        """
        ## Load a single numpy array

        This is used to save processed data
        """
        file_name = name + ".npy"
        return np.load(str(self.info.npy_path / file_name))

    def clear_checkpoints(self):
        """
        ## Clear old checkpoints

        We run this when running a new fresh trial
        """
        path = pathlib.Path(self.info.checkpoint_path)
        if path.exists():
            util.rm_tree(path)

    def clear_summaries(self):
        """
        ## Clear TensorBoard summaries

        We run this when running a new fresh trial
        """
        path = pathlib.Path(self.info.summary_path)
        if path.exists():
            util.rm_tree(path)

    def clear_screenshots(self):
        """
        ## Clear screenshots
        """
        path = pathlib.Path(self.info.screenshots_path)
        if path.exists():
            util.rm_tree(path)

        path.mkdir(parents=True)

    def save_screenshot(self, img, file_name: str):
        """
        ## Save screenshot

        Use this to save images
        """
        img.save(str(self.info.screenshots_path / file_name))

    def _start(self, global_step: int):
        self.trial.start_step = global_step
        logger.set_start_global_step(global_step)

        path = pathlib.Path(self.info.diff_path)
        if not path.exists():
            path.mkdir(parents=True)

        with open(str(path / f"{self.trial.index}.diff"), "w") as f:
            f.write(self.trial.diff)
