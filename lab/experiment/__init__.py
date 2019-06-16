import pathlib
import time
from typing import List, Dict

import numpy as np
import git

from lab import colors, util
from lab.commenter import Commenter
from lab.lab import Lab
from lab.logger import Logger

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

        self.summary_path = self.experiment_path / "log"
        self.screenshots_path = self.experiment_path / 'screenshots'
        self.trials_log_file = self.experiment_path / "trials.yaml"

    def exists(self) -> bool:
        """
        ### Check is this experiment results exists
        """
        p = pathlib.Path(self.summary_path)
        return p.exists()


def _struct_time_to_time(t: time.struct_time):
    return f"{t.tm_hour :02}:{t.tm_min :02}:{t.tm_sec :02}"


def _struct_time_to_date(t: time.struct_time):
    return f"{t.tm_year :04}-{t.tm_mon :02}-{t.tm_mday :02}"


_GLOBAL_STEP = 'global_step'


class Trial:
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

    progress: List[Dict[str, str]]

    def __init__(self, *,
                 python_file: str,
                 trial_date: str,
                 trial_time: str,
                 comment: str,
                 commit: str or None = None,
                 commit_message: str or None = None,
                 is_dirty: bool = True,
                 start_step: int = 0,
                 progress=None,
                 progress_limit: int = 16):
        self.commit = commit
        self.is_dirty = is_dirty
        self.python_file = python_file
        self.trial_date = trial_date
        self.trial_time = trial_time
        self.comment = comment
        self.commit_message = commit_message
        self.start_step = start_step
        if progress is None:
            self.progress = []
        else:
            self.progress = progress

        assert progress_limit % 2 == 0
        assert progress_limit >= 8
        self.progress_limit = progress_limit

    @classmethod
    def new_trial(cls, *,
                  python_file: str,
                  trial_time: time.struct_time,
                  comment: str):
        """
        ## Create a new trial
        """
        return cls(python_file=python_file,
                   trial_date=_struct_time_to_date(trial_time),
                   trial_time=_struct_time_to_time(trial_time),
                   comment=comment)

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        """
        ## Create a new trial from a dictionary
        """
        return cls(**data)

    def to_dict(self):
        """
        ## Convert trial to a dictionary for saving
        """
        return dict(
            python_file=self.python_file,
            trial_date=self.trial_date,
            trial_time=self.trial_time,
            comment=self.comment,
            commit=self.commit,
            commit_message=self.commit_message,
            is_dirty=self.is_dirty,
            start_step=self.start_step,
            progress=self.progress
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

        # Stop if no progress is available
        if len(self.progress) == 0:
            return res

        res.append('')

        # Print progress table
        lens = {}
        for k, v in self.progress[0].items():
            lens[k] = max(len(k), len(v))

        line = []
        for k, v in self.progress[0].items():
            line.append(' ' * (lens[k] - len(k)) + k)
        line = '| ' + ' | '.join(line) + ' |'
        res.append('-' * len(line))
        res.append(line)
        res.append('-' * len(line))

        for p in self.progress:
            line = [' ' * (lens[k] - len(str(v))) + str(v) for k, v in p.items()]
            line = '| ' + ' | '.join(line) + ' |'
            res.append(line)

        res.append('-' * len(line))

        return res

    def __str__(self):
        return f"Trial(comment=\"{self.comment}\"," \
            f" commit=\"{self.commit_message}\"," \
            f" date={self.trial_date}, time={self.trial_time}"

    def __repr__(self):
        return self.__str__()

    def set_progress(self, progress: Dict[str, str]):
        """
        ## Add or update progress
        """

        assert _GLOBAL_STEP in progress

        if len(self.progress) < 2:
            self.progress.append(progress)
        else:
            g0 = int(self.progress[0][_GLOBAL_STEP].replace(',', ''))
            g1 = int(self.progress[1][_GLOBAL_STEP].replace(',', ''))
            gp = int(self.progress[-2][_GLOBAL_STEP].replace(',', ''))
            gf = int(self.progress[-1][_GLOBAL_STEP].replace(',', ''))

            if g1 - g0 <= gf - gp:
                # add
                if len(self.progress) == self.progress_limit:
                    shrunk = []
                    for i in range(self.progress_limit // 2):
                        shrunk.append(self.progress[2 * i + 1])
                    self.progress = shrunk
                self.progress.append(progress)
            else:
                self.progress[-1] = progress


class Experiment:
    """
    ## Experiment

    Each experiment has different configurations or algorithms.
    An experiment can have multiple trials.
    """

    def __init__(self, *,
                 lab: Lab,
                 name: str,
                 python_file: str,
                 comment: str,
                 check_repo_dirty: bool = True):
        """
        ### Create the experiment

        :param lab: reference to current lab
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

        self.__variables = None
        self.info = ExperimentInfo(lab, name)

        self.logger = Logger()
        self.check_repo_dirty = check_repo_dirty

        self.lab = lab

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

    def _log_trial(self, is_add: bool):
        """
        ### Log trial

        This will add or update a trial in the `trials.yaml` file
        """
        try:
            with open(str(self.info.trials_log_file), "r") as file:
                trials = util.yaml_load(file.read())
                if trials is None:
                    trials = []
        except FileNotFoundError:
            trials = []

        if is_add or len(trials) == 0:
            trials.append(self.trial.to_dict())
        else:
            trials[-1] = self.trial.to_dict()

        with open(str(self.info.trials_log_file), "w") as file:
            file.write(util.yaml_dump(trials))

    def _log_python_file(self):
        """
        ### Add header to python source

        This will add or update trial information in python source file
        """

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

    def save_progress(self, progress: Dict[str, str]):
        """
        ## Save experiment progress
        """
        self.trial.set_progress(progress)

        self._log_trial(is_add=False)
        self._log_python_file()
