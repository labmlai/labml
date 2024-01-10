import time
from typing import Dict, List, Optional, Union, NamedTuple

from fastapi import Request

from labml_db import Model, Key, Index, load_keys

from .. import auth
from . import user
from .. import utils
from . import project
from . import computer
from . import status
from .. import settings
from ..analyses.experiments.metrics import MetricsAnalysis, MetricsPreferencesIndex, MetricsPreferencesModel
from ..logger import logger
from .. import analyses
from ..enums import RunEnums

SYNC_INTERVAL = 60 * 20


class CardInfo(NamedTuple):
    class_name: str
    name: str
    is_print: bool
    queue_size: int = 0


class Run(Model['Run']):
    name: str
    owner: str
    comment: str
    note: str
    tags: List[str]
    start_time: float
    run_ip: str
    run_uuid: str
    rank: int
    world_size: int
    python_file: str
    repo_remotes: str
    commit: str
    commit_message: str
    start_step: int
    is_claimed: bool
    status: Key['status.Status']
    configs: Dict[str, any]
    computer_uuid: str
    size: float
    size_checkpoints: float
    size_tensorboard: float
    last_synced: float
    stdout: str
    stdout_unmerged: str
    logger: str
    logger_unmerged: str
    stderr: str
    stderr_unmerged: str
    selected_configs: List['str']
    favourite_configs: List['str']
    main_rank: int

    wildcard_indicators: Dict[str, Dict[str, Union[str, bool]]]
    indicators: Dict[str, Dict[str, Union[str, bool]]]
    errors: List[Dict[str, str]]

    @classmethod
    def defaults(cls):
        return dict(name='',
                    owner='',
                    comment='',
                    note='',
                    tags=[],
                    start_time=None,
                    run_uuid='',
                    rank=0,
                    world_size=0,
                    python_file='',
                    repo_remotes='',
                    commit='',
                    commit_message='',
                    start_step=None,
                    run_ip='',
                    is_claimed=True,
                    status=None,
                    configs={},
                    computer_uuid='',
                    size=None,
                    size_checkpoints=None,
                    size_tensorboard=None,
                    last_synced=None,
                    stdout='',
                    stdout_unmerged='',
                    logger='',
                    logger_unmerged='',
                    stderr='',
                    stderr_unmerged='',
                    wildcard_indicators={},
                    indicators={},
                    errors=[],
                    selected_configs=[],
                    favourite_configs=[],
                    main_rank=0,
                    )

    @property
    def url(self) -> str:
        return f'run/{self.run_uuid}'

    @property
    def is_in_progress(self) -> bool:
        s = get_status(self.run_uuid)

        return s.get_true_status() == RunEnums.RUN_IN_PROGRESS

    @property
    def is_sync_needed(self) -> bool:
        if not self.size_checkpoints or not self.size_tensorboard:
            return True
        else:
            return time.time() - self.last_synced >= SYNC_INTERVAL

    def sync_run(self, **kwargs) -> None:
        size_checkpoints = kwargs.get('size_checkpoints', None)
        size_tensorboard = kwargs.get('size_tensorboard', None)
        size = kwargs.get('size', None)

        if size_checkpoints:
            self.size_checkpoints = size_checkpoints
        if size_tensorboard:
            self.size_tensorboard = size_tensorboard
        if size:
            self.size = size

        self.last_synced = time.time()

        self.save()

    def update_run(self, data: Dict[str, any]) -> None:
        if not self.name:
            self.name = data.get('name', '')
        if not self.comment:
            self.comment = data.get('comment', '')
        if not self.tags:
            self.tags = data.get('tags', [])
        if not self.python_file:
            self.python_file = data.get('python_file', '')
        if not self.repo_remotes:
            self.repo_remotes = data.get('repo_remotes', '')
        if not self.commit:
            self.commit = data.get('commit', '')
        if not self.commit_message:
            self.commit_message = data.get('commit_message', '')
        if self.start_step is None:
            self.start_step = data.get('start_step', '')
        if not self.computer_uuid:
            self.computer_uuid = data.get('computer', '')
            computer.add_run(self.computer_uuid, self.run_uuid)

        if 'configs' in data:
            configs = data.get('configs', {})
            self.configs.update(configs)

            defaults = {}
            for k, v in configs.items():
                computed = v['computed']
                name = v['name']
                if computed and type(computed) == dict and computed.get('type', '') == 'DynamicSchedule':
                    defaults[name] = computed

            if defaults:
                hp_analysis = analyses.AnalysisManager.get_experiment_analysis('HyperParamsAnalysis', self.run_uuid)
                if hp_analysis is not None:
                    hp_analysis.set_default_values(defaults)

        if 'stdout' in data and data['stdout']:
            stdout_processed, self.stdout_unmerged = self.merge_output(self.stdout_unmerged, data['stdout'])
            self.stdout += stdout_processed
        if 'logger' in data and data['logger']:
            logger_processed, self.logger_unmerged = self.merge_output(self.logger_unmerged, data['logger'])
            self.logger += logger_processed
        if 'stderr' in data and data['stderr']:
            stderr_processed, self.stderr_unmerged = self.merge_output(self.stderr_unmerged, data['stderr'])
            self.stderr += stderr_processed

        if 'favorite_configs' in data:
            self.favourite_configs = data.get('favorite_configs', [])
        if 'selected_configs' in data:
            self.selected_configs = data.get('selected_configs', [])

        if not self.indicators:
            self.indicators = data.get('indicators', {})
        if not self.wildcard_indicators:
            self.wildcard_indicators = data.get('wildcard_indicators', {})

        self.save()

    def merge_output(self, unmerged: str, new: str) -> (str, str):
        unmerged += new
        processed = ''
        if len(new) > 1:
            processed, unmerged = self.format_output(unmerged)

        return processed, unmerged

    @staticmethod
    def format_output(output: str) -> (str, str):
        res = []
        temp = ''
        for i, c in enumerate(output):
            if c == '\n':
                temp += '\n'
                res.append(temp)
                temp = ''
            elif c == '\r' and len(output) > i + 1 and output[i + 1] == '\n':
                pass
            elif c == '\r':
                temp = ''
            else:
                temp += c

        return ''.join(res), temp

    @staticmethod
    def format_remote_repo(urls: str) -> str:
        if not urls:
            return ''

        url = urls[0]
        if not url:
            return ''
        if 'git' not in url:
            logger.error(f'unknown repo url: {url}')
            return ''

        split = url.split(':')
        if split[0] != 'https':
            split[0] = 'https'
            return '://github.com/'.join(split)[:-4]

        return url[:-4]

    @staticmethod
    def format_commit(url: str, commit: str) -> str:
        if not url:
            return ''
        if 'unknown' in commit:
            logger.error(f'unknown repo url: {url}, commit:{commit}')
            return 'unknown'

        return url + f'/commit/{commit}'

    def get_rank_uuids(self) -> Dict[int, str]:
        if self.rank == 0 and self.world_size > 1:
            other_rank_run_uuids = \
                {rank: f'{self.run_uuid}_{rank}' if rank != 0 else self.run_uuid for rank in range(self.world_size)}
        else:
            other_rank_run_uuids = {}

        return other_rank_run_uuids

    def get_data(self, request: Request, is_dist_run: bool = False) -> Dict[str, Union[str, any]]:
        u = auth.get_auth_user(request)
        if u:
            is_project_run = u.default_project.is_project_run(self.run_uuid)
        else:
            is_project_run = False

        configs = [{'key': k, **c} for k, c in self.configs.items()]
        formatted_repo = self.format_remote_repo(self.repo_remotes)

        other_rank_run_uuids = self.get_rank_uuids()

        # get the std out and std error from main rank
        stdout = self.stdout + self.stdout_unmerged
        stderr = self.stderr + self.stderr_unmerged
        run_logger = self.logger + self.logger_unmerged

        if self.world_size != 0 and other_rank_run_uuids and is_dist_run:
            run_uuid = other_rank_run_uuids[self.main_rank]
            run = get(run_uuid)
            if run:
                stdout = run.stdout + run.stdout_unmerged
                stderr = run.stderr + run.stderr_unmerged
                run_logger = run.logger + run.logger_unmerged

        return {
            'run_uuid': self.run_uuid,
            'rank': self.rank,
            'other_rank_run_uuids': other_rank_run_uuids,
            'world_size': self.world_size,
            'is_project_run': is_project_run,
            'name': self.name,
            'comment': self.comment,
            'note': self.note,
            'tags': self.tags,
            'start_time': self.start_time,
            'start_step': self.start_step,
            'python_file': self.python_file,
            'repo_remotes': formatted_repo,
            'commit': self.format_commit(formatted_repo, self.commit),
            'commit_message': self.commit_message,
            'is_claimed': self.is_claimed,
            'size': self.size,
            'size_checkpoints': self.size_checkpoints,
            'size_tensorboard': self.size_tensorboard,
            'computer_uuid': self.computer_uuid,
            'configs': configs,
            'stdout': stdout,
            'logger': run_logger,
            'stderr': stderr,
            'favourite_configs': self.favourite_configs,
            'selected_configs': self.selected_configs,
        }

    def get_summary(self) -> Dict[str, str]:
        fav_configs = [{'key': key, **self.configs[key]} for key in self.configs.keys() if
                       key in self.favourite_configs]

        return {
            'run_uuid': self.run_uuid,
            'computer_uuid': self.computer_uuid,
            'name': self.name,
            'comment': self.comment,
            'start_time': self.start_time,
            'world_size': self.world_size,
            'preview_series': None,
            'metric_values': None,
            'favorite_configs': fav_configs,
        }

    def edit_run(self, data: Dict[str, any]) -> None:
        if 'name' in data:
            self.name = data.get('name', self.name)
        if 'comment' in data:
            self.comment = data.get('comment', self.comment)
        if 'note' in data:
            self.note = data.get('note', self.note)

        if 'favourite_configs' in data:
            self.favourite_configs = data.get('favourite_configs', self.favourite_configs)
        if 'selected_configs' in data:
            self.selected_configs = data.get('selected_configs', self.selected_configs)

        if self.world_size > 0:
            run_uuids = [f'{self.run_uuid}_{rank}' for rank in range(1, self.world_size)]
            runs = mget(run_uuids)
            for r in runs:
                r.name = self.name
                r.note = self.note
                r.comment = self.comment
                r.favourite_configs = self.favourite_configs
                r.selected_configs = self.selected_configs
            Run.msave(runs)

        self.save()


class RunIndex(Index['Run']):
    pass


def get_or_create(request: Request, run_uuid: str, rank: int, world_size: int, main_rank: int, labml_token: str = '') -> 'Run':
    p = project.get_project(labml_token)

    if run_uuid in p.runs:
        return p.runs[run_uuid].load()

    run = get(run_uuid)
    if run is not None:
        return run

    if labml_token == settings.FLOAT_PROJECT_TOKEN:
        is_claimed = False
        identifier = ''
    else:
        is_claimed = True

        identifier = user.get_token_owner(labml_token)

    time_now = time.time()

    s = status.create_status()
    run = Run(run_uuid=run_uuid,
              rank=rank,
              world_size=world_size,
              owner=identifier,
              start_time=time_now,
              run_ip=request.client.host,
              is_claimed=is_claimed,
              status=s.key,
              main_rank=main_rank,
              )

    if run.rank == 0:  # TODO
        p.runs[run.run_uuid] = run.key
        p.is_run_added = True

    run.save()
    p.save()

    RunIndex.set(run.run_uuid, run.key)

    return run


def delete(run_uuid: str) -> None:
    r = get(run_uuid)

    if r:
        s = r.status.load()

        computer.remove_run(r.computer_uuid, run_uuid)

        s.delete()
        r.delete()

        RunIndex.delete(run_uuid)

        analyses.AnalysisManager.delete_run(run_uuid)


def get_runs(labml_token: str) -> List['Run']:
    res = []
    p = project.get_project(labml_token)
    load_keys(list(p.runs.values()))

    return res


def get(run_uuid: str) -> Optional['Run']:
    run_key = RunIndex.get(run_uuid)

    if run_key:
        return run_key.load()

    return None


def mget(run_uuids: List[str]) -> List[Optional['Run']]:
    run_keys = RunIndex.mget(run_uuids)
    return load_keys(run_keys)


def get_status(run_uuid: str) -> Union[None, 'status.Status']:
    r = get(run_uuid)

    if r:
        return r.status.load()

    return None


def get_merged_status_data(run_uuids: List[str]) -> Union[None, 'status.Status']:
    r = mget(run_uuids)
    status_keys = [run.status for run in r if run]
    status_list = load_keys(status_keys)
    run_status_keys = [s.run_status for s in status_list if s]
    run_status_list = load_keys(run_status_keys)

    status_data_list = [s.get_data(run_status.to_dict())
                        for s, run_status in zip(status_list, run_status_list) if s and run_status]

    if len(status_data_list) == 0:
        return None

    status_data = status_data_list[0]
    status_data['last_updated_time'] = max([s['last_updated_time'] for s in status_data_list])
    status_data['run_status']['time'] = max([s['run_status']['time'] for s in status_data_list])

    status_priority = {
        RunEnums.RUN_IN_PROGRESS: 1,
        RunEnums.RUN_COMPLETED: 2,
        RunEnums.RUN_CRASHED: 3,
        RunEnums.RUN_INTERRUPTED: 4,
        RunEnums.RUN_NOT_RESPONDING: 5,
        RunEnums.RUN_UNKNOWN: 6,
    }
    status_data['run_status']['status'] = min([s['run_status']['status'] for s in status_data_list],
                                              key=lambda x: status_priority[x])

    return status_data
