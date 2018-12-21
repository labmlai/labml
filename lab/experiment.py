import json
import pathlib
import time
from typing import List, Optional, Dict

import git
import numpy as np
import tensorflow as tf

import lab.colors as colors
from lab import tf_util, util
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
                 progress=None):
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

    def set_progress(self, progress: Dict[str, str], is_add: bool):
        """
        ## Add or update progress
        """
        if is_add:
            self.progress.append(progress)
        else:
            self.progress[-1] = progress


class Experiment:
    """
    ## Experiment

    Each experiment has different configurations or algorithms.
    An experiment can have multiple trials.
    """

    __variables: Optional[List[tf.Variable]]

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

        if not tf.gfile.Exists(str(self.info.experiment_path)):
            tf.gfile.MakeDirs(str(self.info.experiment_path))

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

    @util.deprecated("Use load_checkpoint_numpy")
    def load_checkpoint(self, session: tf.Session):
        """
        ## Load latest TensorFlow checkpoint

        **Use numpy array saving.**

        It's simpler and you can easily load subsets of
        variable.
        Or even manually swap variables between experiments with just
        file copies to try things out.
        """
        if not _load_checkpoint(session, str(self.info.checkpoint_path)):
            tf_util.init_variables(session)
            return False
        else:
            return True

    @util.deprecated("Use save_checkpoint_numpy")
    def save_checkpoint(self, session: tf.Session, global_step: int):
        """
        ## Save TensorFlow checkpoint

        Use numpy array saving.
        """
        _delete_old_checkpoints(str(self.info.checkpoint_path))
        _save_checkpoint(session, str(self.info.checkpoint_path),
                         str(self.info.model_file), global_step)

    def load_checkpoint_numpy(self,
                              session: tf.Session):
        """
        ## Load model as a set of numpy arrays
        """

        checkpoints_path = pathlib.Path(self.info.checkpoint_path)
        max_step = -1
        for c in checkpoints_path.iterdir():
            max_step = max(max_step, int(c.name))

        if max_step < 0:
            return False

        checkpoint_path = checkpoints_path / str(max_step)

        with open(str(checkpoint_path / "info.json"), "r") as f:
            files = json.loads(f.readline())

        # Load each variable
        for variable in self.__variables:
            file_name = files[variable.name]
            value = np.load(str(checkpoint_path / file_name))
            ph = tf.placeholder(value.dtype,
                                shape=value.shape,
                                name=f"{tf_util.strip_variable_name(variable.name)}_ph")

            assign_op = tf.assign(variable, ph)
            session.run(assign_op, feed_dict={ph: value})

        return True

    def save_checkpoint_numpy(self,
                              session: tf.Session,
                              global_step: int):
        """
        ## Save model as a set of numpy arrays
        """

        checkpoints_path = pathlib.Path(self.info.checkpoint_path)
        if not checkpoints_path.exists():
            checkpoints_path.mkdir()

        checkpoint_path = checkpoints_path / str(global_step)
        assert not checkpoint_path.exists()

        checkpoint_path.mkdir()

        values = session.run(self.__variables)

        # Save each variable
        files = {}
        for variable, value in zip(self.__variables, values):
            file_name = tf_util.variable_name_to_file_name(
                tf_util.strip_variable_name(variable.name))
            file_name = f"{file_name}.npy"
            files[variable.name] = file_name

            np.save(str(checkpoint_path / file_name), value)

        # Save header
        with open(str(checkpoint_path / "info.json"), "w") as f:
            f.write(json.dumps(files))

        # Delete old checkpoints
        for c in checkpoints_path.iterdir():
            if c.name != checkpoint_path.name:
                util.rm_tree(c)

    def save_npy(self, array: np.ndarray, name: str):
        """
        ## Save a single numpy array

        This is used to save processed data
        """
        tf.gfile.MkDir(str(self.info.npy_path))
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
        if tf.gfile.Exists(str(self.info.checkpoint_path)):
            tf.gfile.DeleteRecursively(str(self.info.checkpoint_path))

    def clear_summaries(self):
        """
        ## Clear TensorBoard summaries

        We run this when running a new fresh trial
        """
        if tf.gfile.Exists(str(self.info.summary_path)):
            tf.gfile.DeleteRecursively(str(self.info.summary_path))

    def create_writer(self, session: tf.Session):
        """
        ## Create TensorFlow summary writer
        """
        self.logger.writer = tf.summary.FileWriter(str(self.info.summary_path), session.graph)

    def clear_screenshots(self):
        """
        ## Clear screenshots
        """
        path = str(self.info.screenshots_path)
        if tf.gfile.Exists(path):
            tf.gfile.DeleteRecursively(path)

        tf.gfile.MkDir(path)

    def save_screenshot(self, img, file_name: str):
        """
        ## Save screenshot

        Use this to save images
        """
        img.save(str(self.info.screenshots_path / file_name))

    def set_variables(self, variables: List[tf.Variable]):
        """
        ## Set variable for saving and loading
        """
        self.__variables = variables

    def _log_trial(self, is_add: bool):
        """
        ### Log trial

        This will add or update a trial in the `trials.yaml` file
        """
        try:
            with open(str(self.info.trials_log_file), "r") as file:
                trials = util.yaml_load(file.read())
        except FileNotFoundError:
            trials = []

        if is_add:
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
        with open(self.trial.python_file, "r") as file:
            lines = file.read().splitlines()

        trial_print = self.trial.pretty_print()

        lines = commenter.update(lines, trial_print)
        code = '\n'.join(lines)

        with open(self.trial.python_file, "w") as file:
            file.write(code)

    def save_progress(self, progress: Dict[str, str], is_add: bool):
        """
        ## Save experiment progress
        """
        self.trial.set_progress(progress, is_add)

        self._log_trial(is_add=False)
        self._log_python_file()

    def start(self, global_step: int, session: tf.Session):
        """
        ## Start experiment

        Load a checkpoint or reset based on `global_step`.
        """

        self.trial.start_step = global_step
        self._log_trial(is_add=True)
        self._log_python_file()

        if global_step > 0:
            # load checkpoint if we are starting from middle
            with self.logger.monitor("Loading checkpoint") as m:
                if self.__variables is None:
                    m.is_successful = self.load_checkpoint(session)
                else:
                    m.is_successful = self.load_checkpoint_numpy(session)
        else:
            # initialize variables and clear summaries if we are starting from scratch
            with self.logger.monitor("Clearing summaries"):
                self.clear_summaries()
            with self.logger.monitor("Clearing checkpoints"):
                self.clear_checkpoints()
            with self.logger.monitor("Initializing variables"):
                tf_util.init_variables(session)

        self.create_writer(session)


def _delete_old_checkpoints(checkpoint_path: str):
    """
    #### Delete old TensorFlow checkpoints
    """
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if not latest_checkpoint:
        return

    checkpoint_path = pathlib.Path(checkpoint_path)
    for p in checkpoint_path.iterdir():
        if p.match(str(checkpoint_path / 'checkpoint')):
            continue
        elif p.match(latest_checkpoint + '*'):
            continue
        else:
            p.unlink()


def _load_checkpoint(session: tf.Session, checkpoint_path: str) -> bool:
    """
    #### Load TensorFlow checkpoint
    """
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if latest_checkpoint is not None:
        saver = tf.train.Saver()
        saver.restore(session, latest_checkpoint)
        return True
    else:
        return False


def _save_checkpoint(session: tf.Session,
                     checkpoint_path: str,
                     model_file: str,
                     global_step: int):
    """
    #### Save TensorFlow checkpoint
    """
    if not tf.gfile.Exists(checkpoint_path):
        tf.gfile.MakeDirs(checkpoint_path)
    saver = tf.train.Saver()
    saver.save(session, model_file, global_step=global_step)
