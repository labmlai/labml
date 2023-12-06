import os
import time
from pathlib import Path
from typing import List, Dict, Optional

from labml import lab
from labml.internal.configs.processor import load_configs

from .. import util
from ..manage.runs import get_run_by_uuid
from ...logger import Text
from ...utils.notice import labml_notice


def struct_time_to_time(t: time.struct_time):
    return f"{t.tm_hour :02}:{t.tm_min :02}:{t.tm_sec :02}"


def struct_time_to_date(t: time.struct_time):
    return f"{t.tm_year :04}-{t.tm_mon :02}-{t.tm_mday :02}"


_GLOBAL_STEP = 'global_step'
_TAR_BUFFER_SIZE = 100_000


class RunLoadError(TypeError):
    pass


class RunInfo:
    def __init__(self, *,
                 uuid: str,
                 python_file: str,
                 trial_date: str,
                 trial_time: str,
                 name: str,
                 comment: str,
                 commit: Optional[str] = None,
                 commit_message: Optional[str] = None,
                 repo_remotes: List[str] = None,
                 is_dirty: bool = True,
                 experiment_path: Path,
                 start_step: int = 0,
                 notes: str = '',
                 pid: int,
                 load_run: Optional[str] = None,
                 tags: List[str],
                 distributed_rank: int,
                 distributed_world_size: int,
                 distributed_main_rank: int,
                 ):
        self.name = name
        self.uuid = uuid
        if repo_remotes is None:
            repo_remotes = []
        self.repo_remotes = repo_remotes
        self.commit = commit
        self.is_dirty = is_dirty
        self.python_file = python_file
        self.trial_date = trial_date
        self.trial_time = trial_time
        self.comment = comment
        self.commit_message = commit_message
        self.start_step = start_step

        self.load_run = load_run

        self.experiment_path = experiment_path
        self.distributed_rank = distributed_rank
        self.distributed_world_size = distributed_world_size
        self.distributed_main_rank = distributed_main_rank
        self.pid = pid

        if distributed_world_size > 0:
            self.run_path = experiment_path / f'{uuid}.{self.distributed_rank}'
        else:
            self.run_path = experiment_path / str(uuid)

        self.pid_path = self.run_path / 'run.pid'
        self.checkpoint_path = self.run_path / "checkpoints"
        self.numpy_path = self.run_path / "numpy"

        self.diff_path = self.run_path / "source.diff"

        self.artifacts_folder = self.run_path / "artifacts"
        self.log_file = self.run_path / 'log.jsonl'

        self.info_path = self.run_path / "run.yaml"
        self.indicators_path = self.run_path / "indicators.yaml"
        self.configs_path = self.run_path / "configs.yaml"
        self.run_log_path = self.run_path / "run_log.jsonl"
        self.notes = notes
        self.tags = tags

    @classmethod
    def from_path(cls, run_path: Path):
        """
        ## Create a new from path
        """

        if not (run_path / 'run.yaml').exists():
            # TODO
            raise NotImplementedError('Loading distributed sessions')
            # run_info_path = run_path / '0' / 'run.yaml'
        else:
            run_info_path = run_path / 'run.yaml'

        try:
            with open(str(run_info_path), 'r') as f:
                data = util.yaml_load(f.read())
                if not isinstance(data, dict):
                    raise RunLoadError()
                return cls.from_dict(run_path.parent, data)
        except FileNotFoundError:
            raise RunLoadError()

    @classmethod
    def from_dict(cls, experiment_path: Path, data: Dict[str, any]):
        """
        ## Create a new trial from a dictionary
        """
        params = dict(experiment_path=experiment_path)
        params.update(data)
        try:
            run = cls(**params)
            return run
        except TypeError as e:
            raise RunLoadError(e)

    def to_dict(self):
        """
        ## Convert trial to a dictionary for saving
        """
        return dict(
            name=self.name,
            uuid=self.uuid,
            python_file=self.python_file,
            trial_date=self.trial_date,
            trial_time=self.trial_time,
            comment=self.comment,
            repo_remotes=self.repo_remotes,
            commit=self.commit,
            commit_message=self.commit_message,
            is_dirty=self.is_dirty,
            start_step=self.start_step,
            notes=self.notes,
            tags=self.tags,
            distributed_rank=self.distributed_rank,
            distributed_world_size=self.distributed_world_size,
            load_run=self.load_run
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
        return res

    def __str__(self):
        return f"{self.__class__.__name__}(comment=\"{self.comment}\"," \
               f" commit=\"{self.commit_message}\"," \
               f" date={self.trial_date}, time={self.trial_time})"

    def __repr__(self):
        return self.__str__()

    def is_after(self, run: 'Run'):
        if run.trial_date < self.trial_date:
            return True
        elif run.trial_date > self.trial_date:
            return False
        elif run.trial_time < self.trial_time:
            return True
        else:
            return False


class Run(RunInfo):
    """
    # Trial 🏃‍

    Every trial in an experiment has same configs.
    It's just multiple runs.

    A new trial will replace checkpoints
    or previous trials, you should make a copy if needed.
    The performance log in `trials.yaml` is not replaced.

    You should run new trials after bug fixes or to see performance is
     consistent.

    If you want to try different configs, create multiple experiments.
    """

    diff: Optional[str]

    def __init__(self, *,
                 uuid: str,
                 python_file: str,
                 trial_date: str,
                 trial_time: str,
                 name: str,
                 comment: str,
                 repo_remotes: List[str] = None,
                 commit: Optional[str] = None,
                 commit_message: Optional[str] = None,
                 is_dirty: bool = True,
                 experiment_path: Path,
                 start_step: int = 0,
                 notes: str = '',
                 pid: int,
                 tags: List[str],
                 distributed_rank: int,
                 distributed_world_size: int,
                 distributed_main_rank: int
                 ):
        super().__init__(
            python_file=python_file, trial_date=trial_date, trial_time=trial_time,
            name=name, comment=comment, uuid=uuid, experiment_path=experiment_path,
            repo_remotes=repo_remotes,
            commit=commit, commit_message=commit_message, is_dirty=is_dirty,
            start_step=start_step, notes=notes,
            pid=pid,
            tags=tags,
            distributed_rank=distributed_rank, distributed_world_size=distributed_world_size,
            distributed_main_rank=distributed_main_rank,
        )
        self.diff = None

    @classmethod
    def create(cls, *,
               uuid: str,
               experiment_path: Path,
               python_file: str,
               trial_time: time.struct_time,
               name: str,
               comment: str,
               tags: List[str],
               distributed_rank: int,
               distributed_world_size: int,
               distributed_main_rank: int,
               ):
        """
        ## Create a new trial
        """

        return cls(python_file=python_file,
                   trial_date=struct_time_to_date(trial_time),
                   trial_time=struct_time_to_time(trial_time),
                   uuid=uuid,
                   experiment_path=experiment_path,
                   name=name,
                   comment=comment,
                   tags=tags,
                   pid=os.getpid(),
                   distributed_rank=distributed_rank,
                   distributed_world_size=distributed_world_size,
                   distributed_main_rank=distributed_main_rank,
                   )

    def make_path(self):
        run_path = Path(self.run_path)
        if not run_path.exists():
            try:
                run_path.mkdir(parents=True)
            except FileExistsError:
                pass

    def save_info(self):
        self.make_path()

        assert not self.pid_path.exists(), str(self.pid_path)
        assert os.getpid() == self.pid, f'{os.getpid()} != {self.pid}'

        with open(str(self.pid_path), 'w') as f:
            f.write(f'{self.pid}')

        with open(str(self.info_path), "w") as file:
            file.write(util.yaml_dump(self.to_dict()))

        if self.diff is not None:
            with open(str(self.diff_path), "w") as f:
                f.write(self.diff)


def get_configs(run_uuid: str):
    run_path = get_run_by_uuid(lab.get_experiments_path(), run_uuid)
    if run_path is None:
        labml_notice(["Couldn't find a previous run to load configurations: ",
                      (run_uuid, Text.value)], is_danger=True)
        return None

    configs_path = run_path / "configs.yaml"
    configs = load_configs(configs_path)

    return configs
