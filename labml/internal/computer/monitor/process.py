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

    def __init__(self, nvml):
        self.pids = {}
        self.processes = []
        self.data = {}
        self.nvml = nvml

    def _track_gpu(self, idx: int):
        handle = self.nvml.nvmlDeviceGetHandleByIndex(idx)

        procs = self.nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        for p in procs:
            key = self.pids[p.pid]
            self.data.update({
                f'process.{key}.gpu.{idx}.mem': p.usedGpuMemory,
            })

        procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for p in procs:
            key = self.pids[p.pid]
            self.data.update({
                f'process.{key}.gpu.{idx}.mem': p.usedGpuMemory,
            })

    def track_gpus(self):
        self.data = {}

        if not self.nvml:
            return {}

        for i in range(self.nvml.nvmlDeviceGetCount()):
            self._track_gpu(i)

        return self.data

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
                    f'process.{key}.ppid': p.ppid(),
                    f'process.{key}.create_time': p.create_time(),
                })

                try:
                    self.data.update({
                        f'process.{key}.exe': p.exe(),
                    })
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    pass

                try:
                    self.data.update({
                        f'process.{key}.cmdline': '\n'.join(p.cmdline()),
                    })
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            self.processes[key].active = True

            try:
                res = p.memory_info()
                self.data.update({
                    f'process.{key}.rss': res.rss,
                    f'process.{key}.vms': res.vms,
                })
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass

            try:
                res = p.cpu_times()
                self.data.update({
                    f'process.{key}.user': res.user,
                    f'process.{key}.system': res.system,
                })
                if hasattr(res, 'iowait'):
                    self.data.update({
                        f'process.{key}.iowait': res.iowait,
                    })
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass

            try:
                res = p.cpu_percent()
                self.data.update({
                    f'process.{key}.cpu': res,
                })
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass

            try:
                res = p.num_threads()
                self.data.update({
                    f'process.{key}.threads': res,
                })
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass

    def track(self):
        self.data = {}

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

        return self.data