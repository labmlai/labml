import uuid
from typing import Dict, Optional

from starlette.requests import Request

from labml_app.db import project, run
from labml_db import Model, Index


class DistRun(Model['DistRun']):
    uuid: str
    ranks: Dict[int, str]
    main_rank: int
    world_size: int

    @classmethod
    def defaults(cls):
        return dict(uuid='',
                    ranks={},
                    main_rank=-1,
                    world_size=-1)

    def get_or_create_run(self, rank: int, request: Optional['Request'] = None, token: Optional['str'] = None)\
            -> Optional['run.Run']:
        if rank in self.ranks:
            return run.get(self.ranks[rank])

        if self.main_rank == -1 or self.world_size == -1:
            raise RuntimeError('Main rank or world size is not set')

        if request is None:
            raise RuntimeError('Request is required to create a new run')

        if token is None:
            raise RuntimeError('Labml token is required to create a new run')

        r = run.get_or_create(request, str(uuid.uuid4()), rank, self.world_size, self.main_rank, token)
        self.ranks[rank] = r.run_uuid

        self.save()
        return r

    def get_run(self, rank) -> Optional['run.Run']:
        if rank in self.ranks:
            return run.get(self.ranks[rank])
        return None

    def get_main_run(self) -> Optional['run.Run']:
        if self.main_rank == -1:
            return None
        return self.get_run(self.ranks[self.main_rank])

    def get_main_uuid(self) -> str:
        return self.ranks[self.main_rank]

    def delete(self):
        for r_uuid in self.ranks.values():
            run.delete(r_uuid)
        super().delete()


class DistRunIndex(Index['DistRun']):
    pass


def get_or_create(run_uuid: str, labml_token: str):
    p = project.get_project(labml_token)

    if p.is_project_dist_run(run_uuid):
        return p.get_dist_run(run_uuid)
    print(DistRunIndex.get(run_uuid))
    if DistRunIndex.get(run_uuid) is not None:
        raise RuntimeError(f'Dist run {run_uuid} already exists on a different project')

    dr = DistRun()
    dr.uuid = run_uuid
    DistRunIndex.set(run_uuid, dr.key)

    p.add_dist_run_with_model(dr)
    return dr


def get(run_uuid: str) -> Optional['DistRun']:
    key = DistRunIndex.get(run_uuid)
    if key:
        return key.load()
    return None


def mget(run_uuids: list) -> list:
    keys = DistRunIndex.mget(run_uuids)
    return [key.load() for key in keys]


def get_main_run(run_uuid: str) -> Optional['run.Run']:
    dr = get(run_uuid)
    if dr:
        return dr.get_main_run()
    return None


def delete(run_uuid: str):
    dr = get(run_uuid)
    if dr:
        dr.delete()
        DistRunIndex.delete(run_uuid)
