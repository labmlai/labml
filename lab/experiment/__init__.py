import pathlib
import time
from typing import Optional, List, Set

import git
import numpy as np

from lab import colors
from lab import logger
from lab.commenter import Commenter
from lab.experiment.run import Run
from lab.lab import Lab
from lab.logger_class.writers import sqlite, tensorboard

commenter = Commenter(
    comment_start='"""',
    comment_end='"""',
    add_start='```trial',
    add_end='```'
)


class Experiment:
    """
    ## Experiment

    Each experiment has different configurations or algorithms.
    An experiment can have multiple trials.
    """

    # whether not to start the experiment if there are uncommitted changes.
    check_repo_dirty: bool

    def __init__(self, *,
                 name: Optional[str],
                 python_file: Optional[str],
                 comment: Optional[str],
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

        if python_file is None:
            python_file = self.__get_caller_file()

        self.lab = Lab(python_file)

        if name is None:
            file_path = pathlib.PurePath(python_file)
            name = file_path.stem

        if comment is None:
            comment = ''

        self.name = name
        self.experiment_path = self.lab.experiments / name

        self.check_repo_dirty = self.lab.check_repo_dirty

        experiment_path = pathlib.Path(self.experiment_path)
        if not experiment_path.exists():
            experiment_path.mkdir(parents=True)

        self.run = Run.create(
            experiment_path=self.experiment_path,
            python_file=python_file,
            trial_time=time.localtime(),
            comment=comment)

        repo = git.Repo(self.lab.path)

        self.run.commit = repo.head.commit.hexsha
        self.run.commit_message = repo.head.commit.message.strip()
        self.run.is_dirty = repo.is_dirty()
        self.run.diff = repo.git.diff()

        checkpoint_saver = self._create_checkpoint_saver()
        logger.internal.set_checkpoint_saver(checkpoint_saver)

        if writers is None:
            writers = {'sqlite', 'tensorboard'}

        if 'sqlite' in writers:
            logger.internal.add_writer(sqlite.Writer(self.run.sqlite_path))
        if 'tensorboard' in writers:
            logger.internal.add_writer(tensorboard.Writer(self.run.tensorboard_log_path))

    @staticmethod
    def __get_caller_file():
        import inspect

        frames: List[inspect.FrameInfo] = inspect.stack()
        lab_src = pathlib.PurePath(__file__).parent.parent

        for f in frames:
            module_path = pathlib.PurePath(f.filename)
            if str(module_path).startswith(str(lab_src)):
                continue
            return str(module_path)

        return ''

    def _create_checkpoint_saver(self):
        return None

    def print_info_and_check_repo(self):
        """
        ## 🖨 Print the experiment info and check git repo status
        """
        logger.log_color([
            (self.name, colors.Style.bold)
        ])
        logger.log_color([
            ("\t", None),
            (self.run.comment, colors.BrightColor.cyan)
        ])
        logger.log_color([
            ("\t", None),
            ("[dirty]" if self.run.is_dirty else "[clean]", None),
            (": ", None),
            (f"\"{self.run.commit_message.strip()}\"", colors.BrightColor.orange)
        ])

        # Exit if git repository is dirty
        if self.check_repo_dirty and self.run.is_dirty:
            logger.log("Cannot trial an experiment with uncommitted changes. ",
                       new_line=False)
            logger.log("[FAIL]", color=colors.BrightColor.red)
            exit(1)

    def save_npy(self, array: np.ndarray, name: str):
        """
        ## Save a single numpy array

        This is used to save processed data
        """
        npy_path = pathlib.Path(self.run.npy_path)
        npy_path.mkdir(parents=True)
        file_name = name + ".npy"
        np.save(str(self.run.npy_path / file_name), array)

    def load_npy(self, name: str):
        """
        ## Load a single numpy array

        This is used to save processed data
        """
        file_name = name + ".npy"
        return np.load(str(self.run.npy_path / file_name))

    def _start(self, global_step: int):
        self.run.start_step = global_step
        logger.internal.set_start_global_step(global_step)

        self.run.save_info()
        logger.internal.save_indicators(self.run.indicators_path)

        with open(str(self.run.diff_path), "w") as f:
            f.write(self.run.diff)
