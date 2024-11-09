import uuid
from typing import Dict, Optional

from starlette.requests import Request

from labml_app.db import project, run
from labml_db import Model, Index


class DistRun(Model['DistRun']):
    uuid: str
    ranks: Dict[int: str]
    main_rank: int
    world_size: int

    @classmethod
    def defaults(cls):
        return dict(uuid='',
                    ranks={},
                    main_rank=-1,
                    world_size=-1)

    def get_or_create_run(self, rank: int, request: Optional['Request'] = None, token: Optional['str'] = None):
        if rank in self.ranks:
            return run.get(self.ranks[rank])

        if self.main_rank == -1 or self.world_size == -1:
            raise RuntimeError('Main rank or world size is not set')

        if request is None:
            raise RuntimeError('Request is required to create a new run')

        if token is None:
            raise RuntimeError('Labml token is required to create a new run')

        r = run.get_or_create(request, str(uuid.uuid4()), rank, self.world_size, self.main_rank, token)
        self.ranks[rank] = r.uuid

        self.save()
        return r


class DistRunIndex(Index['DistRun']):
    pass


def get_or_create(run_uuid: str, labml_token: str):
    p = project.get_project(labml_token)

    if p.is_project_dist_run(run_uuid):
        return p.get_dist_run(run_uuid)

    if DistRunIndex.get(run_uuid) is not None:
        raise RuntimeError(f'Dist run {run_uuid} already exists on a different project')

    dr = DistRun()
    dr.uuid = run_uuid
    DistRunIndex.set(run_uuid, dr.key)

    p.add_dist_run(run_uuid)


def get(run_uuid: str) -> Optional['DistRun']:
    key = DistRunIndex.get(run_uuid)
    if key:
        return key.load()
    return None
