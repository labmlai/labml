import time
from typing import Dict, List, Optional, Union, NamedTuple

from fastapi import Request

from labml_db import Model, Key, Index

from .. import auth
from . import project
from . import user
from .. import utils
from .. import settings
from . import status


class DistributedRun(Model['DistributedRun']):
    run_uuid: str
    name: str
    comment: str
    owner: str
    world_size: int
    runs: List[str]
    is_claimed: bool
    start_time: float
    status: Key['status.Status']

    @classmethod
    def defaults(cls):
        return dict(run_uuid='',
                    name='',
                    comment='',
                    world_size=0,
                    runs=[],
                    owner='',
                    is_claimed=True,
                    start_time=None,
                    status=None
                    )

    def get_summary(self) -> Dict[str, str]:
        return {
            'run_uuid': self.run_uuid,
            'name': self.name,
            'comment': self.comment,
            'start_time': self.start_time,
        }

    def get_data(self, request: Request) -> Dict[str, Union[str, any]]:
        u = auth.get_auth_user(request)
        if u:
            is_project_run = u.default_project.is_project_run(self.run_uuid)
        else:
            is_project_run = False

        return {
            'run_uuid': self.run_uuid,
            'world_size': self.world_size,
            'is_project_run': is_project_run,
            'name': self.name,
            'comment': self.comment,
            'is_claimed': self.is_claimed,
        }


# TODO handle backend calls, claim

class DistributedRunIndex(Index['DistributedRun']):
    pass


def get_or_create(request: Request, run_uuid: str, world_size: int, labml_token: str = '') -> 'DistributedRun':
    p = project.get_project(labml_token)

    if run_uuid in p.distributed_runs:
        return p.distributed_runs[run_uuid].load()

    if labml_token == settings.FLOAT_PROJECT_TOKEN:
        is_claimed = False
        identifier = ''
    else:
        is_claimed = True

        identifier = user.get_token_owner(labml_token)
        utils.analytics.AnalyticsEvent.track(request, 'run_claimed', {'run_uuid': run_uuid}, identifier=identifier)
        utils.analytics.AnalyticsEvent.run_claimed_set(identifier)

    time_now = time.time()

    s = status.create_status()
    dist_run = DistributedRun(run_uuid=run_uuid,
                              world_size=world_size,
                              owner=identifier,
                              is_claimed=is_claimed,
                              start_time=time_now,
                              status=s.key,
                              )

    p.distributed_runs[dist_run.run_uuid] = dist_run.key
    p.is_run_added = True

    dist_run.save()
    p.save()

    DistributedRunIndex.set(dist_run.run_uuid, dist_run.key)

    return dist_run


def get_runs(labml_token: str) -> List['DistributedRun']:
    res = []
    p = project.get_project(labml_token)
    for run_uuid, run_key in p.distributed_runs.items():
        res.append(run_key.load())

    return res


def get(run_uuid: str) -> Optional['DistributedRun']:
    run_key = DistributedRunIndex.get(run_uuid)

    if run_key:
        return run_key.load()

    return None
