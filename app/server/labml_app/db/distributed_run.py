import time
from typing import Dict, List, Optional, Union, NamedTuple

from fastapi import Request

from labml_db import Model, Key, Index

from . import project
from . import user
from .. import utils
from .. import settings


class DistributedRun(Model['DistributedRun']):
    run_uuid: str
    owner: str
    world_size: int
    runs: List[str]
    is_claimed: bool
    start_time: float

    @classmethod
    def defaults(cls):
        return dict(run_uuid='',
                    world_size=0,
                    runs=[],
                    owner='',
                    is_claimed=True,
                    start_time=None,
                    )


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
    dist_run = DistributedRun(run_uuid=run_uuid,
                              world_size=world_size,
                              owner=identifier,
                              is_claimed=is_claimed,
                              start_time=time_now,
                              )

    p.distributed_runs[dist_run.run_uuid] = dist_run.key
    p.is_run_added = True

    dist_run.save()
    p.save()

    DistributedRunIndex.set(dist_run.run_uuid, dist_run.key)


def get(run_uuid: str) -> Optional['DistributedRun']:
    run_key = DistributedRunIndex.get(run_uuid)

    if run_key:
        return run_key.load()

    return None
