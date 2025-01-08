from typing import List, Dict, Union, Optional, Set

from labml_db import Model, Key, Index

from . import run
from . import session
from ..logger import logger


class Project(Model['Project']):
    labml_token: str
    is_sharable: float
    name: str
    runs: Dict[str, Key['run.Run']]
    sessions: Dict[str, Key['session.Session']]
    is_run_added: bool
    folders: any  # delete from db and then remove
    tag_index: Dict[str, Set[str]]

    @classmethod
    def defaults(cls):
        return dict(name='',
                    is_sharable=False,
                    labml_token='',
                    runs={},
                    sessions={},
                    is_run_added=False,
                    tag_index={},
                    folders={},
                    )

    def is_project_run(self, run_uuid: str) -> bool:
        return run_uuid in self.runs

    def is_project_session(self, session_uuid: str) -> bool:
        return session_uuid in self.sessions

    def _get_runs_util(self, run_uuids: List[str]) -> List['run.Run']:
        res = []

        runs = run.mget(run_uuids)
        run_uuids_from_db = []
        for r in runs:
            if r:
                res.append(r)
                run_uuids_from_db.append(r.run_uuid)

        likely_deleted = set(run_uuids) - set(run_uuids_from_db)
        for run_uuid in likely_deleted:
            if run_uuid in self.runs:
                self.runs.pop(run_uuid)
            for tag, runs in self.tag_index.items():
                if run_uuid in runs:
                    self.tag_index[tag].remove(run_uuid)

        if self.is_run_added:
            self.is_run_added = False

        if not self.is_run_added or likely_deleted:
            self.save()

        return res

    def get_all_tags(self) -> List[str]:
        tags_to_pop = []
        for tag, runs in self.tag_index.items():
            if not runs:
                tags_to_pop.append(tag)

        for tag in tags_to_pop:
            self.tag_index.pop(tag)

        return list(self.tag_index.keys())

    def get_runs(self) -> List['run.Run']:
        run_uuids = list(self.runs.keys())
        return self._get_runs_util(run_uuids)

    def get_runs_by_tags(self, tag: str) -> List['run.Run']:
        if tag in self.tag_index:
            run_uuids = [r for r in self.tag_index[tag]]
            return self._get_runs_util(run_uuids)
        else:
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
            if run_uuid in self.runs:
                r = run.get(run_uuid)
                if r and r.owner == project_owner:
                    try:
                        for tag in r.tags:
                            if tag in self.tag_index:
                                self.tag_index[tag].remove(run_uuid)
                                if len(self.tag_index[tag]) == 0:
                                    self.tag_index.pop(tag)
                        run.delete(run_uuid)
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

    def add_run(self, run_uuid: str) -> None:
        r = run.get(run_uuid)

        if r:
            self.runs[run_uuid] = r.key

            for tag in r.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(run_uuid)

        self.save()

    def add_run_with_model(self, r: run.Run) -> None:
        self.runs[r.run_uuid] = r.key
        self.is_run_added = True

        for tag in r.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(r.run_uuid)

        self.save()

    def edit_run(self, run_uuid: str, data: any):
        r = run.get(run_uuid)
        if r is None:
            raise ValueError(f'Run {run_uuid} not found')

        current_tags = r.tags
        new_tags = data.get('tags', r.tags)

        for tag in current_tags:
            if (tag not in new_tags  # removed tag
                    and tag in self.tag_index
                    and run_uuid in self.tag_index[tag]):
                self.tag_index[tag].remove(run_uuid)

        for tag in new_tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(run_uuid)  # set will handle duplicates

        r.edit_run(data)
        self.save()

    def get_run(self, run_uuid: str) -> Optional['run.Run']:
        if run_uuid in self.runs:
            return self.runs[run_uuid].load()
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


def get_run(run_uuid: str, labml_token: str = '') -> Optional['run.Run']:
    p = get_project(labml_token)

    if p.is_project_run(run_uuid):
        return p.get_run(run_uuid)
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
