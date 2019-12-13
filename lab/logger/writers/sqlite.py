import sqlite3
from pathlib import PurePath
from typing import Dict, Optional

import numpy as np

from . import Writer as WriteBase
from ..indicators import Indicator


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

    def _get_key(self, indicator):
        if indicator.type_ != 'scalar':
            return self._parse_key(f'{indicator.name}.mean')
        else:
            return self._parse_key(f'{indicator.name}')

    def write(self, *,
              global_step: int,
              values: Dict[str, any],
              indicators: Dict[str, Indicator]):
        self.__connect()

        for k, ind in indicators.items():
            v = values[k]
            if len(v) == 0:
                continue
            key = self._get_key(ind)
            self.conn.execute(
                f"INSERT INTO scalars VALUES ('{key}', {global_step}, {float(np.mean(v))})")

        self.conn.commit()
