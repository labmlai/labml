import time
from enum import Enum
from typing import List, Set

from labml_db import Model

from labml_app.db import run


class DefaultFolders(Enum):
    ARCHIVE = 'archive'
    DEFAULT = 'default'


class Folder(Model['Folder']):
    name: str
    created_at: float
    run_uuids: Set[str]

    @classmethod
    def defaults(cls):
        return dict(name='',
                    created_at=time.time(),
                    run_uuids=set(),
                    )

    def get_runs(self) -> List['run.Run']:
        res = []
        for run_uuid in self.run_uuids:
            r_key = run.RunIndex.get(run_uuid)
            if r_key is not None:
                r = r_key.load()
                if r is not None:
                    res.append(r)

        res.sort(key=lambda x: x.start_time, reverse=True)
        return res

    def add_run(self, run_uuid: str):
        self.run_uuids.add(run_uuid)
        self.save()

    def remove_run(self, run_uuid: str):
        self.run_uuids.remove(run_uuid)
        self.save()

