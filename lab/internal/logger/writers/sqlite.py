import sqlite3
from pathlib import PurePath
from typing import Dict, Optional

from . import Writer as WriteBase
from lab.internal.logger.store.artifacts import Artifact
from lab.internal.logger.store.indicators import Indicator


class Writer(WriteBase):
    conn: Optional[sqlite3.Connection]

    def __init__(self, sqlite_path: PurePath):
        super().__init__()

        self.sqlite_path = sqlite_path
        self.conn = None

    def __connect(self):
        if self.conn is not None:
            return

        self.conn = sqlite3.connect(str(self.sqlite_path))

        try:
            self.conn.execute(f"CREATE TABLE scalars "
                              f"(indicator text, step integer, value real)")
            self.conn.execute(f"CREATE TABLE indexed_scalars "
                              f"(indicator text, step integer, idx integer, value real)")

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

        self.conn.commit()
