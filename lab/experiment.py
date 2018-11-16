import json
import pathlib
import time

import git
import tensorflow as tf
import numpy as np

import lab.colors as colors
from lab import tf_util
from lab.lab import Lab
from lab.logger import Logger


class Experiment:
    """
    ## Experiment

    Each run is an experiment.
    """

    def __init__(self, *,
                 lab: Lab,
                 name: str,
                 run_file: str,
                 comment: str,
                 check_repo_dirty: bool = True):
        """
        ### Create the experiment

        :param lab: reference to current lab
        :param name: name of the experiment
        :param run_file: `__file__` that invokes this. This is stored in
         the experiments list.
        :param comment: a short description of the experiment
        :param check_repo_dirty: whether not to start the experiment if
         there are uncommitted changes.

        The experiments log keeps track of `run_file`, `name`, `comment` as
         well as the git commit.

        Experiment maintains the locations of checkpoints, logs, etc.
        """

        self.name = name
        self.logger = Logger()
        self.check_repo_dirty = check_repo_dirty

        self.lab = lab
        self.experiment_path = lab.experiments / name

        if not tf.gfile.Exists(str(self.experiment_path)):
            tf.gfile.MakeDirs(str(self.experiment_path))

        checkpoint_path = self.experiment_path / "checkpoints"
        self.checkpoint_path = str(checkpoint_path)
        self.npy_path = self.experiment_path / "npy"
        self.model_file = str(checkpoint_path / 'model')

        self.summary_path = str(self.experiment_path / "log")
        self.screenshots_path = self.experiment_path / 'screenshots'

        if not self._log_run(run_file, comment):
            self.logger.log("Cannot run an experiment with uncommitted changes. ", new_line=False)
            self.logger.log("[FAIL]", color=colors.BrightColor.red)
            exit(1)

    def _log_run(self, run_file, comment):
        repo = git.Repo(self.lab.path)
        if self.check_repo_dirty and repo.is_dirty():
            return False

        log_file = str(self.experiment_path / "runs.txt")
        t = time.localtime()

        with open(log_file, "a+") as file:
            data = {
                'commit': repo.active_branch.commit.hexsha,
                'run_file': run_file,
                'date': "{}-{}-{}".format(t.tm_year, t.tm_mon, t.tm_mday),
                "time": "{}:{}:{}".format(t.tm_hour, t.tm_min, t.tm_sec),
                "comment": comment,
                'commit_message': repo.active_branch.commit.message
            }

            file.write("{}\n".format(json.dumps(data)))

        return True

    def load_checkpoint(self, session: tf.Session):
        """
        Load latest TensorFlow checkpoint
        """
        if not _load_checkpoint(session, self.checkpoint_path):
            tf_util.init_variables(session)
            return False
        else:
            return True

    def save_checkpoint(self, session: tf.Session, global_step: int):
        """
        Save TensorFlow checkpoint
        """
        _delete_old_checkpoints(self.checkpoint_path)
        _save_checkpoint(session, self.checkpoint_path, self.model_file, global_step)

    def save_npy(self, array: np.ndarray, name: str):
        """
        Save numpy array
        """
        tf.gfile.MkDir(str(self.npy_path))
        file_name = name + ".npy"
        np.save(str(self.npy_path / file_name), array)

    def load_npy(self, name: str):
        """
        Load numpy array
        """
        file_name = name + ".npy"
        return np.load(str(self.npy_path / file_name))

    def clear_checkpoints(self):
        """
        Clear old checkpoints
        """
        if tf.gfile.Exists(self.checkpoint_path):
            tf.gfile.DeleteRecursively(self.checkpoint_path)

    def clear_summaries(self):
        """
        Clear TensorBoard summaries
        """
        if tf.gfile.Exists(self.summary_path):
            tf.gfile.DeleteRecursively(self.summary_path)

    def create_writer(self, session: tf.Session):
        """
        Create TensorFlow summary writer
        """
        self.logger.writer = tf.summary.FileWriter(self.summary_path, session.graph)

    def clear_screenshots(self):
        """
        Clear screenshots
        """
        path = str(self.screenshots_path)
        if tf.gfile.Exists(path):
            tf.gfile.DeleteRecursively(path)

        tf.gfile.MkDir(path)

    def save_screenshot(self, img, file_name: str):
        """
        Save screenshot
        """
        img.save(str(self.screenshots_path / file_name))

    def start(self, global_step: int, session: tf.Session):
        """
        Start by either by loading a checkpoint or resetting.
        """
        if global_step > 0:
            # load checkpoint if we are starting from middle
            with self.logger.monitor("Loading checkpoint") as m:
                m.is_successful = self.load_checkpoint(session)
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
    #### Load model
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
    #### Save model
    """
    if not tf.gfile.Exists(checkpoint_path):
        tf.gfile.MakeDirs(checkpoint_path)
    saver = tf.train.Saver()
    saver.save(session, model_file, global_step=global_step)
