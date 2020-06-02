import sqlite3
from typing import Optional

import numpy as np

from .analytics import Analytics, BASIS_POINTS


def _filter_steps(start_step: Optional[int], end_step: Optional[int]):
    sql = ''
    if start_step is not None:
        sql += f' AND step >= {start_step}'
    if end_step is not None:
        sql += f' AND step < {end_step}'

    return sql


class SQLiteAnalytics(Analytics):
    def __init__(self, sqlite_path):
        self.conn = sqlite3.connect(str(sqlite_path))

    def get_key(self, name):
        return name

    def scalar(self, name: str, start_step: Optional[int], end_step: Optional[int]):
        key = self.get_key(name)
        sql = f'SELECT step, value from scalars WHERE indicator = "{key}"'
        sql += _filter_steps(start_step, end_step)
        cur = self.conn.execute(sql)
        return [c for c in cur]

    def summarize(self, events):
        step = np.mean([e[0] for e in events])
        values = np.sort([e[1] for e in events])
        basis_points = np.percentile(values, BASIS_POINTS)

        return np.concatenate(([step], basis_points))

    def tensor(self, name: str, start_step: Optional[int], end_step: Optional[int]):
        key = self.get_key(name)
        sql = f'SELECT step, filename from tensors WHERE indicator = "{key}"'
        sql += _filter_steps(start_step, end_step)
        cur = self.conn.execute(sql)
        return [c for c in cur]
