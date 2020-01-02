import sqlite3
from pathlib import PurePath
from typing import Dict, Optional

from . import Writer as WriteBase
from ..indicators import Indicator, Scalar


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
            self.conn.execute(f"CREATE TABLE scalars (indicator text, step integer, value real)")

        except sqlite3.OperationalError:
            print('Scalar table exists')

    @staticmethod
    def _parse_key(key: str):
        return key
        # if we name tables
        # return key.replace('.', '_')

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        self.__connect()

        for ind in indicators.values():
            if ind.is_empty():
                continue
            v = ind.get_mean()
            if v is None:
                continue

            key = self._parse_key(ind.mean_key)
            self.conn.execute(
                f"INSERT INTO scalars VALUES ('{key}', {global_step}, {v})")

        self.conn.commit()
