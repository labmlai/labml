import time
from pathlib import Path

import psutil

from labml import monit


def watch_jobs(path: Path):
    for job in path.iterdir():
        pid_file = job / 'job.pid'
        if not pid_file.exists():
            continue

        stopped_file = job / 'stopped'
        if stopped_file.exists():
            continue

        with open(str(pid_file), 'r') as f:
            content = f.read().strip()
            pid = int(content)

        if not psutil.pid_exists(pid):
            with open(str(stopped_file), 'w') as f:
                f.write('not running')


def main():
    path = Path('.') / '.jobs'
    while True:
        with monit.section('Watch job status'):
            watch_jobs(path)
        time.sleep(5)


if __name__ == '__main__':
    main()
