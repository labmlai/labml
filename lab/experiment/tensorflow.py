import json
import pathlib
from typing import List, Optional

import numpy as np
import tensorflow as tf

from lab import tf_util, util, experiment
from lab.lab import Lab
from lab.logger import tensorboard_writer


class Experiment(experiment.Experiment):
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

        super().__init__(lab=lab,
                         name=name,
                         python_file=python_file,
                         comment=comment,
                         check_repo_dirty=check_repo_dirty)

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

    def create_writer(self, session: tf.Session):
        """
        ## Create TensorFlow summary writer
        """
        self.logger.add_writer(tensorboard_writer.Writer(
            tf.summary.FileWriter(str(self.info.summary_path), session.graph)))

    def set_variables(self, variables: List[tf.Variable]):
        """
        ## Set variable for saving and loading
        """
        self.__variables = variables

    def start_train(self, global_step: int, session: tf.Session):
        """
        ## Start experiment

        Load a checkpoint or reset based on `global_step`.
        """

        self.trial.start_step = global_step
        self._init_trial_log()

        if global_step > 0:
            # load checkpoint if we are starting from middle
            with self.logger.section("Loading checkpoint") as m:
                m.is_successful = self.load_checkpoint_numpy(session)
        else:
            # initialize variables and clear summaries if we are starting from scratch
            with self.logger.section("Clearing summaries"):
                self.clear_summaries()
            with self.logger.section("Clearing checkpoints"):
                self.clear_checkpoints()
            with self.logger.section("Initializing variables"):
                tf_util.init_variables(session)

        self.create_writer(session)

    def start_replay(self, session: tf.Session):
        """
        ## Start replaying experiment

        Load a checkpoint or reset based on `global_step`.
        """

        with self.logger.section("Loading checkpoint") as m:
            m.is_successful = self.load_checkpoint_numpy(session)
