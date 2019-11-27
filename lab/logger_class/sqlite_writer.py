import sqlite3
from pathlib import Path

import numpy as np

import lab.logger_class.writers

_HISTOGRAM_QUANTILES_10 = [i / 10. for i in range(11)]


class Writer(lab.logger_class.writers.Writer):
    def __init__(self, sqlite_path: Path):
        super().__init__()

        self.conn = sqlite3.connect(str(sqlite_path))

        try:
            # Create table
            self.conn.execute('''CREATE TABLE scalars
                         (key text, step integer, value real)''')
        except sqlite3.OperationalError:
            print('Table exists')

    def write(self, *, global_step: int,
              queues,
              histograms,
              pairs,
              scalars):
        for k, v in queues.items():
            if len(v) == 0:
                continue
            # summary.value.add(tag=k, histo=_get_histogram(v))
            self.conn.execute(
                f"INSERT INTO scalars VALUES ('{k}_mean', {global_step}, {float(np.mean(v))})")

        for k, v in histograms.items():
            if len(v) == 0:
                continue
            # summary.value.add(tag=k, histo=_get_histogram(v))
            self.conn.execute(
                f"INSERT INTO scalars VALUES ('{k}_mean', {global_step}, {float(np.mean(v))})")

        # for k, v in pairs.items():
        #     if len(v) == 0:
        #         continue
        #     summary.value.add(tag=k, tensor=_get_pair_histogram(v))

        for k, v in scalars.items():
            if len(v) == 0:
                continue
            self.conn.execute(
                f"INSERT INTO scalars VALUES ('{k}', {global_step}, {float(np.mean(v))})")

        self.conn.commit()
