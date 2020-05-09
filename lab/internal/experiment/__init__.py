import json
import os
import pathlib
import time
from typing import Optional, List, Set, Dict, Union

import git

from lab import logger, monit
from lab.internal.configs import Configs, ConfigProcessor
from lab.internal.experiment.experiment_run import Run
from lab.internal.lab import lab_singleton
from lab.internal.logger import logger_singleton as logger_internal
from lab.internal.util import is_ipynb
from lab.logger import Text
from lab.utils import get_caller_file


class CheckpointSaver:
    def save(self, global_step):
        raise NotImplementedError()

    def load(self, checkpoint_path):
        raise NotImplementedError()


class Checkpoint(CheckpointSaver):
    _models: Dict[str, any]

    def __init__(self, path: pathlib.PurePath):
        self.path = path
        self._models = {}

    def add_models(self, models: Dict[str, any]):
        """
        ## Set variable for saving and loading
        """
        self._models.update(models)

    def save_model(self,
                   name: str,
                   model: any,
                   checkpoint_path: pathlib.Path) -> any:
        raise NotImplementedError()

    def save(self, global_step):
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
        for name, model in self._models.items():
            files[name] = self.save_model(name, model, checkpoint_path)

        # Save header
        with open(str(checkpoint_path / "info.json"), "w") as f:
            f.write(json.dumps(files))

    def load_model(self,
                   name: str,
                   model: any,
                   checkpoint_path: pathlib.Path,
                   info: any):
        raise NotImplementedError()

    def load(self, checkpoint_path):
        """
        ## Load model as a set of numpy arrays
        """

        with open(str(checkpoint_path / "info.json"), "r") as f:
            files = json.loads(f.readline())

        # Load each model
        for name, model in self._models.items():
            self.load_model(name, model, checkpoint_path, files[name])

        return True


class Experiment:
    r"""
    Each experiment has different configurations or algorithms.
    An experiment can have multiple runs.

    Keyword Arguments:
        name (str, optional): name of the experiment
        python_file (str, optional): path of the Python file that
            created the experiment
        comment (str, optional): a short description of the experiment
        writers (Set[str], optional): list of writers to write stat to
        ignore_callers: (Set[str], optional): list of files to ignore when
            automatically determining ``python_file``
        tags (Set[str], optional): Set of tags for experiment
    """

    run: Run
    configs_processor: Optional[ConfigProcessor]

    # whether not to start the experiment if there are uncommitted changes.
    check_repo_dirty: bool
    checkpoint_saver: CheckpointSaver

    def __init__(self, *,
                 name: Optional[str],
                 python_file: Optional[str],
                 comment: Optional[str],
                 writers: Set[str],
                 ignore_callers: Set[str],
                 tags: Optional[Set[str]]):
        if python_file is None:
            python_file = get_caller_file(ignore_callers)

        if python_file.startswith('<ipython'):
            assert is_ipynb()
            if name is None:
                raise ValueError("You must specify python_file or experiment name"
                                 " when creating an experiment from a python notebook.")

            lab_singleton().set_path(os.getcwd())
            python_file = 'notebook.ipynb'
        else:
            lab_singleton().set_path(python_file)

            if name is None:
                file_path = pathlib.PurePath(python_file)
                name = file_path.stem

        if comment is None:
            comment = ''

        self.name = name
        self.experiment_path = lab_singleton().experiments / name

        self.check_repo_dirty = lab_singleton().check_repo_dirty

        self.configs_processor = None

        experiment_path = pathlib.Path(self.experiment_path)
        if not experiment_path.exists():
            experiment_path.mkdir(parents=True)

        if tags is None:
            tags = set(name.split('_'))

        self.run = Run.create(
            experiment_path=self.experiment_path,
            python_file=python_file,
            trial_time=time.localtime(),
            comment=comment,
            tags=list(tags))

        repo = git.Repo(lab_singleton().path)

        self.run.commit = repo.head.commit.hexsha
        self.run.commit_message = repo.head.commit.message.strip()
        self.run.is_dirty = repo.is_dirty()
        self.run.diff = repo.git.diff()

        logger_internal().reset_writers()

        if 'sqlite' in writers:
            from lab.internal.logger.writers import sqlite
            logger_internal().add_writer(sqlite.Writer(self.run.sqlite_path))
        if 'tensorboard' in writers:
            from lab.internal.logger.writers import tensorboard
            logger_internal().add_writer(tensorboard.Writer(self.run.tensorboard_log_path))

        self.checkpoint_saver = None

    def __print_info_and_check_repo(self):
        """
        🖨 Print the experiment info and check git repo status
        """

        logger.log()
        logger.log([
            (self.name, Text.title),
            ': ',
            (str(self.run.uuid), Text.meta)
        ])

        if self.run.comment != '':
            logger.log(['\t', (self.run.comment, Text.highlight)])

        logger.log([
            "\t"
            "[dirty]" if self.run.is_dirty else "[clean]",
            ": ",
            (f"\"{self.run.commit_message.strip()}\"", Text.highlight)
        ])

        if self.run.load_run is not None:
            logger.log([
                "\t"
                "loaded from",
                ": ",
                (f"{self.run.load_run}", Text.meta2),
            ])

        # Exit if git repository is dirty
        if self.check_repo_dirty and self.run.is_dirty:
            logger.log([("[FAIL]", Text.danger),
                        " Cannot trial an experiment with uncommitted changes."])
            exit(1)

    def _load_checkpoint(self, checkpoint_path: pathlib.PurePath):
        if self.checkpoint_saver is not None:
            self.checkpoint_saver.load(checkpoint_path)

    def save_checkpoint(self):
        if self.checkpoint_saver is not None:
            self.checkpoint_saver.save(logger_internal().global_step)

    def calc_configs(self,
                     configs: Optional[Configs],
                     configs_dict: Dict[str, any],
                     run_order: Optional[List[Union[List[str], str]]]):
        self.configs_processor = ConfigProcessor(configs, configs_dict)
        self.configs_processor(run_order)

        logger.log()

    def __start_from_checkpoint(self, run_uuid: str, checkpoint: Optional[int]):
        checkpoint_path, global_step = experiment_run.get_last_run_checkpoint(
            self.experiment_path,
            run_uuid,
            checkpoint)

        if global_step is None:
            return 0
        else:
            with monit.section("Loading checkpoint"):
                self._load_checkpoint(checkpoint_path)
            self.run.load_run = run_uuid

        return global_step

    def start(self, *,
              run_uuid: Optional[str] = None,
              checkpoint: Optional[int] = None):
        if run_uuid is not None:
            if checkpoint is None:
                checkpoint = -1
            global_step = self.__start_from_checkpoint(run_uuid, checkpoint)
        else:
            global_step = 0

        self.run.start_step = global_step
        logger_internal().set_start_global_step(global_step)

        self.__print_info_and_check_repo()
        if self.configs_processor is not None:
            self.configs_processor.print()

        self.run.save_info()

        if self.configs_processor is not None:
            self.configs_processor.save(self.run.configs_path)

        logger_internal().save_indicators(self.run.indicators_path)
        logger_internal().save_artifacts(self.run.artifacts_path)
        if self.configs_processor:
            logger_internal().write_h_parameters(self.configs_processor.get_hyperparams())


_internal: Optional[Experiment] = None


def experiment_singleton() -> Experiment:
    global _internal

    assert _internal is not None

    return _internal


def create_experiment(*,
                      name: Optional[str],
                      python_file: Optional[str],
                      comment: Optional[str],
                      writers: Set[str],
                      ignore_callers: Set[str],
                      tags: Optional[Set[str]]):
    global _internal

    _internal = Experiment(name=name,
                           python_file=python_file,
                           comment=comment,
                           writers=writers,
                           ignore_callers=ignore_callers,
                           tags=tags)
