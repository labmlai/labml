import sqlite3
import time
from pathlib import PurePath, Path
from typing import Dict, Optional

import numpy as np
from ..store.indicators import Indicator
from ..store.indicators.artifacts import Tensor
from ..store.indicators.indexed import IndexedIndicator
from ..store.indicators.numeric import NumericIndicator

from . import Writer as WriteBase


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

    def _write_indicator(self, global_step: int, indicator: Indicator):
        if indicator.is_empty():
            return

        if isinstance(indicator, NumericIndicator):
            value = indicator.get_mean()
            key = self._parse_key(indicator.mean_key)
            self.conn.execute(
                f"INSERT INTO scalars VALUES (?, ?, ?)",
                (key, global_step, value))

        if isinstance(indicator, IndexedIndicator):
            idx, value = indicator.get_index_mean()
            key = self._parse_key(indicator.mean_key)
            data = [(key, global_step, i, v) for i, v in zip(idx, value)]
            self.conn.executemany(
                f"INSERT INTO indexed_scalars VALUES (?, ?, ?, ?)",
                data)

        if isinstance(indicator, Tensor):
            key = self._parse_key(indicator.name)
            for k in indicator.keys():
                tensor = indicator.get_value(k)
                if not indicator.is_once:
                    filename = f'{key}_{global_step}_{k}.npy'
                    self.conn.execute(
                        f"INSERT INTO tensors VALUES (?, ?, ?)",
                        (key, global_step, filename))
                else:
                    filename = f'{key}_{k}.npy'
                self.conn.execute(
                    f"INSERT INTO tensors VALUES (?, ?, ?)",
                    (key, -1, filename))

                artifacts_folder = Path(self.artifacts_path)
                if not artifacts_folder.exists():
                    artifacts_folder.mkdir(parents=True)

                np.save(str(self.artifacts_path / filename), tensor)

    def write(self, *,
              global_step: int,
              indicators: Dict[str, Indicator]):
        self.__connect()

        for ind in indicators.values():
            self._write_indicator(global_step, ind)

        t = time.time()
        if t - self.last_committed > 0.1:
            self.last_committed = t
            self.flush()

    def flush(self):
        if self.conn is not None:
            self.conn.commit()
