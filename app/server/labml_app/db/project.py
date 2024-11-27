from typing import List, Dict, Union, Optional, Set

from labml_db import Model, Key, Index

from . import run, dist_run
from . import session
from .dist_run import DistRunIndex
from ..logger import logger


class Project(Model['Project']):
    labml_token: str
    is_sharable: float
    name: str
    dist_runs: Dict[str, Key['dist_run.DistRun']]
    sessions: Dict[str, Key['session.Session']]
    is_run_added: bool
    folders: any  # delete from db and then remove
    tag_index: Dict[str, Set[str]]
    dist_tag_index: Dict[str, Set[str]]
    runs: Dict  # delete from db and then remove

    @classmethod
    def defaults(cls):
        return dict(name='',
                    is_sharable=False,
                    labml_token='',
                    dist_runs={},
                    sessions={},
                    is_run_added=False,
                    tag_index={},
                    dist_tag_index={},
                    )

    def is_project_dist_run(self, uuid: str) -> bool:
        return uuid in self.dist_runs

    def is_project_session(self, session_uuid: str) -> bool:
        return session_uuid in self.sessions

    def _get_dist_run_util(self, uuids: List[str]) -> List['run.Run']:
        d_runs = dist_run.mget(uuids)

        run_uuids = [dr.get_main_uuid() for dr in d_runs if dr is not None]
        runs = run.mget(run_uuids)

        res = []
        run_uuids_from_db = []
        for r, r_uuid, dist_uuid in zip(runs, run_uuids, uuids):
            if r:
                r.run_uuid = dist_uuid
                res.append(r)
                run_uuids_from_db.append(r_uuid)

        likely_deleted = set(run_uuids) - set(run_uuids_from_db)
        for run_uuid in likely_deleted:
            if run_uuid in self.dist_runs:
                self.dist_runs.pop(run_uuid)
            for tag, runs in self.dist_tag_index.items():
                if run_uuid in runs:
                    self.dist_tag_index[tag].remove(run_uuid)

        if self.is_run_added:
            self.is_run_added = False

        if not self.is_run_added or likely_deleted:
            self.save()

        return res

    def get_dist_runs(self) -> List['run.Run']:
        dist_run_uuids = list(self.dist_runs.keys())
        return self._get_dist_run_util(dist_run_uuids)

    def get_dist_run_by_tags(self, tag: str) -> List['run.Run']:
        if tag in self.dist_tag_index:
            run_uuids = [r for r in self.dist_tag_index[tag]]
            return self._get_dist_run_util(run_uuids)
        return []

    def get_sessions(self) -> List['session.Session']:
        res = []
        for session_uuid, session_key in self.sessions.items():
            session_data = session_key.load()
            if session_data is not None:
                res.append(session_data)

        return res

    def delete_runs(self, run_uuids: List[str], project_owner: str) -> None:
        for run_uuid in run_uuids:
            if run_uuid in self.dist_runs:
                r = dist_run.get_main_run(run_uuid)
                if r and r.owner == project_owner:
                    try:
                        for tag in r.tags:
                            if tag in self.dist_tag_index and run_uuid in self.dist_tag_index[tag]:
                                self.dist_tag_index[tag].remove(run_uuid)
                        dist_run.delete(run_uuid)
                        self.dist_runs.pop(run_uuid)
                        DistRunIndex.delete(run_uuid)
                    except TypeError:
                        logger.error(f'error while deleting the run {run_uuid}')

        self.save()

    def delete_sessions(self, session_uuids: List[str], project_owner: str) -> None:
        for session_uuid in session_uuids:
            if session_uuid in self.sessions:
                self.sessions.pop(session_uuid)
                s = session.get(session_uuid)
                if s and s.owner == project_owner:
                    try:
                        session.delete(session_uuid)
                    except TypeError:
                        logger.error(f'error while deleting the session {session_uuid}')

        self.save()

    def add_dist_run_with_model(self, dr: dist_run.DistRun):
        self.dist_runs[dr.uuid] = dr.key
        self.is_run_added = True

        r = dr.get_main_run()
        if r is not None:
            for tag in r.tags:
                if tag not in self.dist_tag_index:
                    self.dist_tag_index[tag] = set()
                self.dist_tag_index[tag].add(dr.uuid)

        self.save()

    def edit_run(self, dist_run_uuid: str, data: any):
        r = dist_run.get_main_run(dist_run_uuid)
        if r is None:
            raise ValueError(f'Main rank under {dist_run_uuid} not found')

        current_tags = r.tags
        new_tags = data.get('tags', r.tags)

        for tag in current_tags:
            if (tag not in new_tags  # removed tag
                    and tag in self.dist_tag_index
                    and dist_run_uuid in self.dist_tag_index[tag]):
                self.dist_tag_index[tag].remove(dist_run_uuid)

        for tag in new_tags:
            if tag not in self.dist_tag_index:
                self.dist_tag_index[tag] = set()
            self.dist_tag_index[tag].add(dist_run_uuid)  # set will handle duplicates

        r.edit_run(data)
        self.save()

    def get_dist_run(self, uuid: str) -> Optional['dist_run.DistRun']:
        if uuid in self.dist_runs:
            return self.dist_runs[uuid].load()
        else:
            return None

    def add_session(self, session_uuid: str) -> None:
        s = session.get(session_uuid)

        if s:
            self.sessions[session_uuid] = s.key

        self.save()


class ProjectIndex(Index['Project']):
    pass


def get_project(labml_token: str) -> Union[None, Project]:
    project_key = ProjectIndex.get(labml_token)

    if project_key:
        return project_key.load()

    return None


def get_dist_run(run_uuid: str, labml_token: str = '') -> Optional['dist_run.DistRun']:
    p = get_project(labml_token)

    if p.is_project_dist_run(run_uuid):
        return p.get_dist_run(run_uuid)
    else:
        return None


def get_session(session_uuid: str, labml_token: str = '') -> Optional['session.Session']:
    p = get_project(labml_token)

    if session_uuid in p.sessions:
        return p.sessions[session_uuid].load()
    else:
        return None


def create_project(labml_token: str, name: str) -> None:
    project_key = ProjectIndex.get(labml_token)

    if not project_key:
        project = Project(labml_token=labml_token,
                          name=name,
                          is_sharable=True
                          )
        ProjectIndex.set(project.labml_token, project.key)
        project.save()
