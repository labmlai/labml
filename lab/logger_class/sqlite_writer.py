import sqlite3
from pathlib import PurePath
from typing import Dict, Optional

import numpy as np

import lab.logger_class.writers
from .indicators import Indicator


class Writer(lab.logger_class.writers.Writer):
    conn: Optional[sqlite3.Connection]

    def __init__(self, sqlite_path: PurePath):
        super().__init__()

        self.sqlite_path = sqlite_path
        self.conn = None

    def __connect(self, indicators: Dict[str, Indicator]):
        if self.conn is not None:
            return

        self.conn = sqlite3.connect(str(self.sqlite_path))

        try:
            for k, ind in indicators.items():
                if ind.type_ != 'scalar':
                    key = self._parse_key(f'{ind.name}.mean')
                else:
                    key = self._parse_key(f'{ind.name}')
                self.conn.execute(f"CREATE TABLE values_{key} (step integer, value real)")

        except sqlite3.OperationalError:
            print('Scalar table exists')

    @staticmethod
    def _parse_key(key: str):
        return key.replace('.', '_')

    def write(self, *,
              global_step: int,
              values: Dict[str, any],
              indicators: Dict[str, Indicator]):
        self.__connect(indicators)

        for k, ind in indicators.items():
            v = values[k]
            if len(v) == 0:
                continue
            # summary.value.add(tag=k, histo=_get_histogram(v))
            if ind.type_ != 'scalar':
                key = self._parse_key(f'{ind.name}.mean')
            else:
                key = self._parse_key(f'{ind.name}')
            self.conn.execute(
                f"INSERT INTO values_{key} VALUES ({global_step}, {float(np.mean(v))})")

        self.conn.commit()
