import json
import os
import pathlib
import time
import warnings
from typing import Optional, List, Set, Dict, Union

import git

from labml import logger, monit
from labml.internal.configs.base import Configs
from labml.internal.configs.processor import ConfigProcessor
from labml.internal.configs.processor_dict import ConfigProcessorDict
from labml.internal.experiment.experiment_run import Run
from labml.internal.lab import lab_singleton
from labml.internal.logger import logger_singleton as logger_internal
from labml.internal.util import is_ipynb, is_colab, is_kaggle
from labml.logger import Text
from labml.utils import get_caller_file


class ModelSaver:
    def save(self, checkpoint_path: pathlib.Path) -> any:
        raise NotImplementedError()

    def load(self, checkpoint_path: pathlib.Path, info: any):
        raise NotImplementedError()


class CheckpointSaver:
    model_savers: Dict[str, ModelSaver]

    def __init__(self, path: pathlib.PurePath):
        self.path = path
        self.model_savers = {}

    def add_savers(self, models: Dict[str, ModelSaver]):
        """
        ## Set variable for saving and loading
        """
        self.model_savers.update(models)

    def save(self, global_step):
        """
        ## Save model as a set of numpy arrays
        """

        if not self.model_savers:
            return

        checkpoints_path = pathlib.Path(self.path)
        if not checkpoints_path.exists():
            checkpoints_path.mkdir()

        checkpoint_path = checkpoints_path / str(global_step)
        assert not checkpoint_path.exists()

        checkpoint_path.mkdir()

        info = {}
        for name, saver in self.model_savers.items():
            info[name] = saver.save(checkpoint_path)

        # Save header
        with open(str(checkpoint_path / "info.json"), "w") as f:
            f.write(json.dumps(info))

    def load(self, checkpoint_path: pathlib.Path, models: List[str] = None):
        """
        ## Load model as a set of numpy arrays
        """

        if not self.model_savers:
            return False

        if not models:
            models = list(self.model_savers.keys())

        with open(str(checkpoint_path / "info.json"), "r") as f:
            info = json.loads(f.readline())

        # Load each model
        for name in models:
            saver = self.model_savers[name]
            saver.load(checkpoint_path, info[name])

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
        if global_params_singleton().comment is not None:
            comment = global_params_singleton().comment

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

        try:
            repo = git.Repo(lab_singleton().path)

            self.run.commit = repo.head.commit.hexsha
            self.run.commit_message = repo.head.commit.message.strip()
            self.run.is_dirty = repo.is_dirty()
            self.run.diff = repo.git.diff()
        except git.InvalidGitRepositoryError:
            if not is_colab() and not is_kaggle():
                warnings.warn(f"Not a valid git repository",
                              UserWarning, stacklevel=4)
            self.run.commit = 'unknown'
            self.run.commit_message = ''
            self.run.is_dirty = True
            self.run.diff = ''

        logger_internal().reset_writers()

        if 'sqlite' in writers:
            from labml.internal.logger.writers import sqlite
            logger_internal().add_writer(
                sqlite.Writer(self.run.sqlite_path, self.run.artifacts_folder))
        if 'tensorboard' in writers:
            from labml.internal.logger.writers import tensorboard
            logger_internal().add_writer(tensorboard.Writer(self.run.tensorboard_log_path))
        if 'web_api' in writers:
            from labml.internal.logger.writers import web_api
            self.web_api = web_api.Writer()
            logger_internal().add_writer(self.web_api)
        else:
            self.web_api = None

        self.checkpoint_saver = CheckpointSaver(self.run.checkpoint_path)

    def __print_info(self):
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

    def _load_checkpoint(self, checkpoint_path: pathlib.Path):
        if not self.checkpoint_saver.load(checkpoint_path):
            logger.log('No models registered', Text.warning)

    def save_checkpoint(self):
        self.checkpoint_saver.save(logger_internal().global_step)

    def calc_configs(self,
                     configs: Configs,
                     configs_override: Optional[Dict[str, any]],
                     run_order: Optional[List[Union[List[str], str]]]):
        if configs_override is None:
            configs_override = {}
        if global_params_singleton().configs is not None:
            configs_override.update(global_params_singleton().configs)

        self.configs_processor = ConfigProcessor(configs, configs_override)
        self.configs_processor(run_order)

        logger.log()

    def calc_configs_dict(self,
                          configs: Dict[str, any],
                          configs_override: Optional[Dict[str, any]]):
        self.configs_processor = ConfigProcessorDict(configs, configs_override)
        self.configs_processor()

        logger.log()

    def __start_from_checkpoint(self, run_uuid: str, checkpoint: Optional[int]):
        checkpoint_path, global_step = experiment_run.get_run_checkpoint(
            run_uuid,
            checkpoint)

        if global_step is None:
            return 0
        else:
            with monit.section("Loading checkpoint"):
                self._load_checkpoint(checkpoint_path)
            self.run.load_run = run_uuid

        return global_step

    def load_models(self, *,
                    models: List[str],
                    run_uuid: Optional[str] = None,
                    checkpoint: Optional[int] = None):
        if checkpoint is None:
            checkpoint = -1
        checkpoint_path, global_step = experiment_run.get_run_checkpoint(
            run_uuid,
            checkpoint)

        if global_step is None:
            warnings.warn(f"Could not find checkpoint",
                          UserWarning, stacklevel=4)
            return

        with monit.section("Loading checkpoint"):
            _ = self.checkpoint_saver.load(checkpoint_path, models)

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

        self.__print_info()
        if self.check_repo_dirty and self.run.is_dirty:
            logger.log([("[FAIL]", Text.danger),
                        " Cannot trial an experiment with uncommitted changes."])
            exit(1)

        if self.configs_processor is not None:
            self.configs_processor.print()

        self.run.save_info()

        if self.configs_processor is not None:
            self.configs_processor.save(self.run.configs_path)

        if self.web_api is not None:
            self.web_api.set_info(run_uuid=self.run.uuid,
                                  name=self.name,
                                  comment=self.run.comment)
            if self.configs_processor is not None:
                self.web_api.set_configs(self.configs_processor.to_json())
            self.web_api.start()

        logger_internal().save_indicators(self.run.indicators_path)
        if self.configs_processor:
            logger_internal().write_h_parameters(self.configs_processor.get_hyperparams())


class GlobalParams:
    def __init__(self):
        self.configs = None
        self.comment = None


_global_params: Optional[GlobalParams] = None
_internal: Optional[Experiment] = None


def global_params_singleton() -> GlobalParams:
    global _global_params

    if _global_params is None:
        _global_params = GlobalParams()

    return _global_params


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
