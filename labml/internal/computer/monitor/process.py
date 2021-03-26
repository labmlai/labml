from typing import Dict, List

import psutil


class ProcessInfo:
    key: int
    pid: int
    name: str

    def __init__(self, key: int, pid: int, name: str):
        self.key = key
        self.pid = pid
        self.name = name

        self.alive = True
        self.active = True


class ProcessMonitor:
    pids: Dict[int, int]
    processes: List[ProcessInfo]

    def __init__(self):
        self.pids = {}
        self.processes = []
        self.data = {}

    def track_process(self, p: psutil.Process):
        with p.oneshot():
            key = None
            if p.pid in self.pids:
                key = self.pids[p.pid]
                if self.processes[key].name != p.name():
                    key = None
            if key is None:
                key = len(self.processes)
                self.processes.append(ProcessInfo(key,
                                               p.pid,
                                               p.name()))
                self.pids[p.pid] = key
                self.data.update({
                    f'process.{key}.name': p.name(),
                    f'process.{key}.pid': p.pid,
                })

            self.processes[key].active = True

            try:
                res = p.memory_info()
                self.data.update({
                    f'process.{key}.rss': res.rss,
                    f'process.{key}.vms': res.vms,
                })
            except psutil.AccessDenied:
                pass
            try:
                res = p.memory_percent()
                self.data.update({
                    f'process.{key}.mem': res,
                })
            except psutil.AccessDenied:
                pass

            try:
                res = p.cpu_percent()
                self.data.update({
                    f'process.{key}.cpu': res,
                })
            except psutil.AccessDenied:
                pass

            try:
                res = p.num_threads()
                self.data.update({
                    f'process.{key}.threads': res,
                })
            except psutil.AccessDenied:
                pass

    def track(self, data):
        self.data = data

        for p in self.processes:
            p.active = False

        for p in psutil.process_iter():
            self.track_process(p)

        for p in self.processes:
            if not p.active and p.alive:
                p.alive = False
                self.data.update({
                    f'process.{p.key}.dead': True,
                })

