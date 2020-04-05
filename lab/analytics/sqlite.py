import sqlite3

import numpy as np

from .analytics import Analytics, BASIS_POINTS


class SQLiteAnalytics(Analytics):
    def __init__(self, sqlite_path):
        self.conn = sqlite3.connect(str(sqlite_path))

    def get_key(self, name):
        return name

    def scalar(self, name):
        key = self.get_key(name)
        cur = self.conn.execute(
            f'SELECT step, value from scalars WHERE indicator = "{key}"')
        return [c for c in cur]

    def summarize(self, events):
        step = np.mean([e[0] for e in events])
        values = np.sort([e[1] for e in events])
        basis_points = np.percentile(values, BASIS_POINTS)

        return np.concatenate(([step], basis_points))
