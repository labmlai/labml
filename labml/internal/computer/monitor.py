import time
from pathlib import Path
from typing import NamedTuple, Dict

import psutil

from labml.internal.api import ApiCaller
from labml.internal.computer.configs import computer_singleton
from labml.internal.computer.writer import Writer, Header
from labml.utils.notice import labml_notice


class Process(NamedTuple):
    key: int
    pid: int
    name: str


class MonitorComputer:
    processes: Dict[int, Process]

    def __init__(self, session_uuid: str, open_browser):
        api_caller = ApiCaller(computer_singleton().web_api.url,
                               {'computer_uuid': computer_singleton().uuid, 'session_uuid': session_uuid},
                               timeout_seconds=15,
                               daemon=True)
        self.writer = Writer(api_caller, frequency=computer_singleton().web_api.frequency)
        self.header = Header(api_caller,
                             frequency=computer_singleton().web_api.frequency,
                             open_browser=open_browser)
        self.data = {}
        self.cache = {}
        self.nvml = None
        self.n_gpu = 0

        self.processes = {}
        self.n_processes = 0

        try:
            from py3nvml import py3nvml as nvml
            self.nvml = nvml
        except ImportError:
            labml_notice('Install py3nvml to monitor GPUs:\n pip install py3nvml',
                         is_warn=False)

    def start(self):
        configs = {
            'os': self.get_os(),
            'cpu.logical': psutil.cpu_count(),
            'cpu.physical': psutil.cpu_count(logical=False),
        }

        configs.update(self.get_gpu_header())

        self.header.start(configs)

        self.first()

    def track_gpu(self):
        if not self.nvml:
            return

        self.nvml.nvmlInit()
        for i in range(self.n_gpu):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
            self.data.update({
                f'gpu.memory.used.{i}': self.nvml.nvmlDeviceGetMemoryInfo(handle).used,
                f'gpu.utilization.{i}': self.nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                f'gpu.temperature.{i}': self.nvml.nvmlDeviceGetTemperature(handle, self.nvml.NVML_TEMPERATURE_GPU),
                f'gpu.power.usage.{i}': self.nvml.nvmlDeviceGetPowerUsage(handle),
            })

        self.nvml.nvmlShutdown()

    def first_gpu(self):
        if not self.nvml:
            return

        self.nvml.nvmlInit()
        for i in range(self.n_gpu):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
            self.data.update({
                f'gpu.memory.total.{i}': self.nvml.nvmlDeviceGetMemoryInfo(handle).total,
                f'gpu.power.limit.{i}': self.nvml.nvmlDeviceGetPowerManagementLimit(handle),
            })

        self.nvml.nvmlShutdown()

    def get_gpu_header(self):
        if not self.nvml:
            return {}

        self.nvml.nvmlInit()
        self.n_gpu = self.nvml.nvmlDeviceGetCount()
        res = {'gpus': self.n_gpu}
        for i in range(self.n_gpu):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
            res.update({
                f'gpu.name.{i}': self.nvml.nvmlDeviceGetName(handle),
            })
        self.nvml.nvmlShutdown()

        return res

    def track_net_io_counters(self):
        res = psutil.net_io_counters()
        t = time.time()
        if 'net.recv' in self.cache:
            td = t - self.cache['net.time']
            self.data.update({
                'net.recv': (res.bytes_recv - self.cache['net.recv']) / td,
                'net.sent': (res.bytes_sent - self.cache['net.sent']) / td,
            })
        self.cache['net.recv'] = res.bytes_recv
        self.cache['net.sent'] = res.bytes_sent
        self.cache['net.time'] = t

    def track_memory(self):
        res = psutil.virtual_memory()
        self.data.update({
            'memory.total': res.total,
            'memory.used': res.used,
            'memory.available': res.available,
        })

    def track_cpu(self):
        res = psutil.cpu_times()
        self.data.update({
            'cpu.idle': res.idle,
            'cpu.system': res.system,
            'cpu.user': res.user,
        })
        res = psutil.cpu_freq()
        if res is not None:
            self.data.update({
                'cpu.freq': res.current,
                'cpu.freq.min': res.min,
                'cpu.freq.max': res.max,
            })
        res = psutil.cpu_percent(percpu=True)
        self.data.update({f'cpu.perc.{i}': p for i, p in enumerate(res)})

    def track_disk(self):
        res = psutil.disk_usage(Path.home())
        self.data.update({
            'disk.free': res.free,
            'disk.total': res.total,
            'disk.used': res.used,
        })

    def track_process(self, p: psutil.Process):
        with p.oneshot():
            key = None
            if p.pid in self.processes:
                if self.processes[p.pid].name == p.name():
                    key = self.processes[p.pid].key
            if key is None:
                key = self.n_processes
                self.n_processes += 1

                self.processes[p.pid] = Process(key,
                                                p.pid,
                                                p.name())
                self.data.update({
                    f'process.{key}.name': p.name(),
                    f'process.{key}.pid': p.pid,
                })

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

    def track_processes(self):
        for p in psutil.process_iter():
            self.track_process(p)

    def track(self):
        self.track_net_io_counters()
        # inspect(psutil.net_if_addrs())
        # inspect(psutil.net_if_stats())
        self.track_memory()
        self.track_cpu()
        self.track_disk()
        self.track_gpu()
        self.track_processes()

        self.writer.track(self.data)
        self.data = {}

    def first(self):
        # self.track_memory()
        # self.track_cpu()
        # self.track_disk()
        self.first_gpu()
        # track_processes()

        self.writer.track(self.data)
        self.data = {}

    @staticmethod
    def get_os():
        if psutil.MACOS:
            return 'macos'
        elif psutil.LINUX:
            return 'linux'
        elif psutil.WINDOWS:
            return 'windows'
        else:
            return 'unknown'
