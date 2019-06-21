import pathlib
import time
from typing import Dict, Optional

import git
import numpy as np

from lab import colors, util
from lab.commenter import Commenter
from lab.experiment.experiment_trial import Trial
from lab.lab import Lab
from lab.logger import Logger, ProgressSaver

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
        self.screenshots_path = self.experiment_path / 'screenshots'
        self.trials_log_file = self.experiment_path / "trials.yaml"

    def exists(self) -> bool:
        """
        ### Check is this experiment results exists
        """
        p = pathlib.Path(self.summary_path)
        return p.exists()


class _ExperimentProgressSaver(ProgressSaver):
    def __init__(self, *,
                 trial: Trial,
                 trials_log_file: pathlib.PurePath,
                 is_log_python_file: bool):
        self.trial = trial
        self.trials_log_file = trials_log_file
        self.is_log_python_file = is_log_python_file

    def __log_python_file(self):
        if not self.is_log_python_file:
            return

        try:
            with open(self.trial.python_file, "r") as file:
                lines = file.read().splitlines()

            trial_print = self.trial.pretty_print()

            lines = commenter.update(lines, trial_print)
            code = '\n'.join(lines)

            with open(self.trial.python_file, "w") as file:
                file.write(code)
        except FileNotFoundError:
            pass

    def __log_trial(self, is_add: bool):
        """
        ### Log trial

        This will add or update a trial in the `trials.yaml` file
        """
        try:
            with open(str(self.trials_log_file), "r") as file:
                trials = util.yaml_load(file.read())
                if trials is None:
                    trials = []
        except FileNotFoundError:
            trials = []

        if is_add or len(trials) == 0:
            trials.append(self.trial.to_dict())
        else:
            trials[-1] = self.trial.to_dict()

        self.trial.index = len(trials) - 1

        with open(str(self.trials_log_file), "w") as file:
            file.write(util.yaml_dump(trials))

    def save(self, progress: Optional[Dict[str, str]] = None):
        if progress is not None:
            self.trial.set_progress(progress)
            self.__log_trial(is_add=False)
        else:
            self.__log_trial(is_add=True)

        self.__log_python_file()


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

        self.trial.commit = repo.active_branch.commit.hexsha
        self.trial.commit_message = repo.active_branch.commit.message.strip()
        self.trial.is_dirty = repo.is_dirty()
        self.trial.diff = repo.git.diff()
        self.__progress_saver = _ExperimentProgressSaver(trial=self.trial,
                                                         trials_log_file=self.info.trials_log_file,
                                                         is_log_python_file=is_log_python_file)
        self.logger = Logger(progress_saver=self.__progress_saver)

    def print_info_and_check_repo(self):
        """
        ## 🖨 Print the experiment info and check git repo status
        """
        self.logger.log_color([
            (self.info.name, colors.Style.bold)
        ])
        self.logger.log_color([
            ("\t", None),
            (self.trial.comment, colors.BrightColor.cyan)
        ])
        self.logger.log_color([
            ("\t", None),
            ("[dirty]" if self.trial.is_dirty else "[clean]", None),
            (": ", None),
            (f"\"{self.trial.commit_message.strip()}\"", colors.BrightColor.orange)
        ])

        # Exit if git repository is dirty
        if self.check_repo_dirty and self.trial.is_dirty:
            self.logger.log("Cannot trial an experiment with uncommitted changes. ",
                            new_line=False)
            self.logger.log("[FAIL]", color=colors.BrightColor.red)
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

    def _start(self):
        self.__progress_saver.save()

        path = pathlib.Path(self.info.diff_path)
        if not path.exists():
            path.mkdir(parents=True)

        with open(str(path / f"{self.trial.index}.diff"), "w") as f:
            f.write(self.trial.diff)