import sqlite3
import time
from pathlib import PurePath
from typing import Dict, Optional

import numpy as np

from . import Writer as WriteBase
from ..store.artifacts import Artifact, Tensor
from ..store.indicators import Indicator


class Writer(WriteBase):
    conn: Optional[sqlite3.Connection]

    def __init__(self, sqlite_path: PurePath, artifacts_path: PurePath):
        super().__init__()

        self.sqlite_path = sqlite_path
        self.artifacts_path = artifacts_path
        self.conn = None
        self.scalars_cache = []
        self.indexed_scalars_cache = []
        self.last_committed = time.time()

    def __connect(self):
        if self.conn is not None:
            return

        self.conn = sqlite3.connect(str(self.sqlite_path))

        try:
            self.conn.execute(f"CREATE TABLE scalars "
                              f"(indicator text, step integer, value real)")
            self.conn.execute(f"CREATE TABLE indexed_scalars "
                              f"(indicator text, step integer, idx integer, value real)")
            self.conn.execute(f"CREATE TABLE tensors "
                              f"(indicator text, step integer, filename text)")

        except sqlite3.OperationalError:
            print('Scalar table exists')

    @staticmethod
    def _parse_key(key: str):
        return key
        # if we name tables
        # return key.replace('.', '_')

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator],
              artifacts: Dict[str, Artifact]):
        self.__connect()

        for ind in indicators.values():
            if ind.is_empty():
                continue

            value = ind.get_mean()
            if value is not None:
                key = self._parse_key(ind.mean_key)
                self.conn.execute(
                    f"INSERT INTO scalars VALUES (?, ?, ?)",
                    (key, global_step, value))

            idx, value = ind.get_index_mean()
            if idx is not None:
                key = self._parse_key(ind.mean_key)
                data = [(key, global_step, i, v) for i, v in zip(idx, value)]
                self.conn.executemany(
                    f"INSERT INTO indexed_scalars VALUES (?, ?, ?, ?)",
                    data)

        for art in artifacts.values():
            if art.is_empty():
                continue
            key = self._parse_key(art.name)
            if isinstance(art, Tensor):
                for k in art.keys():
                    tensor = art.get_value(k)
                    if not art.is_once:
                        filename = f'{key}_{global_step}_{k}.npy'
                        self.conn.execute(
                            f"INSERT INTO tensors VALUES (?, ?, ?)",
                            (key, global_step, filename))
                    else:
                        filename = f'{key}_{k}.npy'
                    self.conn.execute(
                        f"INSERT INTO tensors VALUES (?, ?, ?)",
                        (key, -1, filename))
                    np.save(str(self.artifacts_path / filename), tensor)

        t = time.time()
        if t - self.last_committed > 0.1:
            self.last_committed = t
            self.flush()

    def flush(self):
        if self.conn is not None:
            self.conn.commit()
