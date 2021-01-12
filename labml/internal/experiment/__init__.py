import json
import os
import pathlib
import time
from typing import Optional, List, Set, Dict, Union, TYPE_CHECKING

import git
from labml import logger, monit
from labml.internal.configs.base import Configs
from labml.internal.configs.processor import ConfigProcessor, FileConfigsSaver
from labml.internal.experiment.experiment_run import Run, struct_time_to_time, struct_time_to_date
from labml.internal.experiment.watcher import ExperimentWatcher
from labml.internal.lab import lab_singleton
from labml.internal.monitor import monitor_singleton as monitor
from labml.internal.tracker import tracker_singleton as tracker
from labml.internal.util import is_ipynb, is_colab, is_kaggle
from labml.logger import Text
from labml.utils import get_caller_file
from labml.utils.notice import labml_notice

if TYPE_CHECKING:
    from labml.internal.api.experiment import ApiExperiment


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
        self.__no_savers_warned = False

    def add_savers(self, models: Dict[str, ModelSaver]):
        """
        ## Set variable for saving and loading
        """
        if experiment_singleton().is_started:
            raise RuntimeError('Cannot register models with the experiment after experiment has started.'
                               'Register models before calling experiment.start')

        self.model_savers.update(models)

    def save(self, global_step):
        """
        ## Save model as a set of numpy arrays
        """

        if not self.model_savers:
            if not self.__no_savers_warned:
                labml_notice(["No models were registered for saving\n",
                              "You can register models with ",
                              ('experiment.add_pytorch_models', Text.value)])
                self.__no_savers_warned = True
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
            if not self.__no_savers_warned:
                labml_notice(["No models were registered for loading or saving\n",
                              "You can register models with ",
                              ('experiment.add_pytorch_models', Text.value)])
                self.__no_savers_warned = True
            return

        if not models:
            models = list(self.model_savers.keys())

        with open(str(checkpoint_path / "info.json"), "r") as f:
            info = json.loads(f.readline())

        to_load = []
        not_loaded = []
        missing = []
        for name in models:
            if name not in info:
                missing.append(name)
            else:
                to_load.append(name)
        for name in info:
            if name not in models:
                not_loaded.append(name)

        # Load each model
        for name in to_load:
            saver = self.model_savers[name]
            saver.load(checkpoint_path, info[name])

        if missing:
            labml_notice([(f'{missing} ', Text.highlight),
                          ('model(s) could not be found.\n'),
                          (f'{to_load} ', Text.none),
                          ('models were loaded.', Text.none)
                          ], is_danger=True)
        if not_loaded:
            labml_notice([(f'{not_loaded} ', Text.none),
                          ('models were not loaded.\n', Text.none),
                          'Models to be loaded should be specified with: ',
                          ('experiment.add_pytorch_models', Text.value)])


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
    web_api: Optional['ApiExperiment']
    is_started: bool
    run: Run
    configs_processor: Optional[ConfigProcessor]

    # whether not to start the experiment if there are uncommitted changes.
    check_repo_dirty: bool
    checkpoint_saver: CheckpointSaver

    distributed_rank: int
    distributed_world_size: int

    def __init__(self, *,
                 uuid: str,
                 name: Optional[str],
                 python_file: Optional[str],
                 comment: Optional[str],
                 writers: Set[str],
                 ignore_callers: Set[str],
                 tags: Optional[Set[str]],
                 is_evaluate: bool):

        if is_ipynb():
            lab_singleton().set_path(os.getcwd())
            if python_file is None:
                python_file = 'notebook.ipynb'
            if name is None:
                name = 'Notebook Experiment'
        else:
            if python_file is None:
                python_file = get_caller_file(ignore_callers)

            lab_singleton().set_path(python_file)

            if name is None:
                file_path = pathlib.PurePath(python_file)
                name = file_path.stem

        if comment is None:
            comment = ''
        if global_params_singleton().comment is not None:
            comment = global_params_singleton().comment

        self.experiment_path = lab_singleton().experiments / name

        self.check_repo_dirty = lab_singleton().check_repo_dirty

        self.configs_processor = None

        if tags is None:
            tags = set(name.split('_'))

        self.run = Run.create(
            uuid=uuid,
            experiment_path=self.experiment_path,
            python_file=python_file,
            trial_time=time.localtime(),
            name=name,
            comment=comment,
            tags=list(tags))

        try:
            repo = git.Repo(lab_singleton().path)

            self.run.repo_remotes = list(repo.remote().urls)
            self.run.commit = repo.head.commit.hexsha
            self.run.commit_message = repo.head.commit.message.strip()
            self.run.is_dirty = repo.is_dirty()
            self.run.diff = repo.git.diff()
        except git.InvalidGitRepositoryError:
            if not is_colab() and not is_kaggle():
                labml_notice(["Not a valid git repository: ",
                              (lab_singleton().path, Text.value)])
            self.run.commit = 'unknown'
            self.run.commit_message = ''
            self.run.is_dirty = True
            self.run.diff = ''

        self.checkpoint_saver = CheckpointSaver(self.run.checkpoint_path)
        self.is_evaluate = is_evaluate
        self.web_api = None
        self.writers = writers
        self.is_started = False
        self.distributed_rank = 0
        self.distributed_world_size = -1

    def __print_info(self):
        """
        🖨 Print the experiment info and check git repo status
        """

        logger.log()
        logger.log([
            (self.run.name, Text.title),
            ': ',
            (str(self.run.uuid), Text.meta)
        ])

        if self.run.comment != '':
            logger.log(['\t', (self.run.comment, Text.highlight)])

        commit_message = self.run.commit_message.strip().replace('\n', '¶ ').replace('\r', '')
        logger.log([
            "\t"
            "[dirty]" if self.run.is_dirty else "[clean]",
            ": ",
            (f"\"{commit_message}\"", Text.highlight)
        ])

        if self.run.load_run is not None:
            logger.log([
                "\t"
                "loaded from",
                ": ",
                (f"{self.run.load_run}", Text.meta2),
            ])

    def _load_checkpoint(self, checkpoint_path: pathlib.Path):
        self.checkpoint_saver.load(checkpoint_path)

    def save_checkpoint(self):
        if self.is_evaluate:
            return
        if self.distributed_rank != 0:
            return

        self.checkpoint_saver.save(tracker().global_step)

    def calc_configs(self,
                     configs: Union[Configs, Dict[str, any]],
                     configs_override: Optional[Dict[str, any]]):
        if configs_override is None:
            configs_override = {}
        if global_params_singleton().configs is not None:
            configs_override.update(global_params_singleton().configs)

        self.configs_processor = ConfigProcessor(configs, configs_override)

        if self.distributed_rank == 0:
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
        checkpoint_path, global_step = experiment_run.get_run_checkpoint(run_uuid, checkpoint)

        if global_step is None:
            labml_notice(['Could not find saved checkpoint'], is_danger=True)
            return

        with monit.section("Loading checkpoint"):
            self.checkpoint_saver.load(checkpoint_path, models)

    def _save_pid(self):
        if not self.run.pids_path.exists():
            self.run.pids_path.mkdir()

        pid_path = self.run.pids_path / f'{self.distributed_rank}.pid'
        assert not pid_path.exists()

        with open(str(pid_path), 'w') as f:
            f.write(f'{os.getpid()}')

    def distributed(self, rank: int, world_size: int):
        self.distributed_rank = rank
        self.distributed_world_size = world_size

        if self.distributed_rank != 0:
            monitor().silent()

        # to make sure we have the path to save pid
        self.run.make_path()

    def _start_tracker(self):
        tracker().reset_writers()

        if self.is_evaluate:
            return

        if self.distributed_rank != 0:
            return

        if 'screen' in self.writers:
            from labml.internal.tracker.writers import screen
            tracker().add_writer(screen.ScreenWriter())

        if 'sqlite' in self.writers:
            from labml.internal.tracker.writers import sqlite
            tracker().add_writer(sqlite.Writer(self.run.sqlite_path, self.run.artifacts_folder))

        if 'tensorboard' in self.writers:
            from labml.internal.tracker.writers import tensorboard
            tracker().add_writer(tensorboard.Writer(self.run.tensorboard_log_path))

        if 'file' in self.writers:
            from labml.internal.tracker.writers import file
            tracker().add_writer(file.Writer(self.run.log_file))

        if 'web_api' in self.writers:
            web_api_conf = lab_singleton().web_api
            if web_api_conf is not None:
                from labml.internal.tracker.writers import web_api
                from labml.internal.api import ApiCaller
                from labml.internal.api.experiment import ApiExperiment
                api_caller = ApiCaller(web_api_conf.url,
                                       {'run_uuid': self.run.uuid},
                                       timeout_seconds=15)
                self.web_api = ApiExperiment(api_caller,
                                             frequency=web_api_conf.frequency,
                                             open_browser=web_api_conf.open_browser)
                tracker().add_writer(web_api.Writer(api_caller,
                                                    frequency=web_api_conf.frequency))
        else:
            self.web_api = None

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

        self._start_tracker()
        tracker().set_start_global_step(global_step)

        if self.distributed_rank == 0:
            self.__print_info()
            if self.check_repo_dirty and self.run.is_dirty:
                logger.log([("[FAIL]", Text.danger),
                            " Cannot trial an experiment with uncommitted changes."])
                exit(1)

        if not self.is_evaluate:
            if self.distributed_rank == 0:
                self.run.save_info()
            self._save_pid()

            if self.distributed_rank == 0:
                if self.configs_processor is not None:
                    self.configs_processor.add_saver(FileConfigsSaver(self.run.configs_path))

                if self.web_api is not None:
                    self.web_api.start(self.run)
                    if self.configs_processor is not None:
                        self.configs_processor.add_saver(self.web_api.get_configs_saver())

                tracker().save_indicators(self.run.indicators_path)

                # PERF: Writing to tensorboard takes about 4 seconds
                # Also wont work when configs are updated live
                # if self.configs_processor:
                #     tracker().write_h_parameters(self.configs_processor.get_hyperparams())

        self.is_started = True
        return ExperimentWatcher(self)

    def finish(self, status: str, details: any = None):
        if not self.is_evaluate:
            with open(str(self.run.run_log_path), 'a') as f:
                end_time = time.time()
                data = json.dumps({'status': status,
                                   'rank': self.distributed_rank,
                                   'details': details,
                                   'time': end_time}, indent=None)
                f.write(data + '\n')

        tracker().finish_loop()

        if self.web_api is not None:
            self.web_api.status(self.distributed_rank, status, details, end_time)


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


def has_experiment() -> bool:
    global _internal

    return _internal is not None


def experiment_singleton() -> Experiment:
    global _internal

    if _internal is None:
        raise RuntimeError('Experiment not created. '
                           'Create an experiment first with `experiment.create`'
                           ' or `experiment.record`')

    return _internal


def create_experiment(*,
                      uuid: str,
                      name: Optional[str],
                      python_file: Optional[str],
                      comment: Optional[str],
                      writers: Set[str],
                      ignore_callers: Set[str],
                      tags: Optional[Set[str]],
                      is_evaluate: bool):
    global _internal

    _internal = Experiment(uuid=uuid,
                           name=name,
                           python_file=python_file,
                           comment=comment,
                           writers=writers,
                           ignore_callers=ignore_callers,
                           tags=tags,
                           is_evaluate=is_evaluate)
