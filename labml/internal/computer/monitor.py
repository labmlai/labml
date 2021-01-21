import time
from pathlib import Path
from typing import Dict

import psutil

from labml.internal.api import ApiCaller
from labml.internal.computer.configs import computer_singleton
from labml.internal.computer.writer import Writer, Header


class MonitorComputer:
    def __init__(self, session_uuid: str):
        api_caller = ApiCaller(computer_singleton().web_api.url,
                               {'computer_uuid': computer_singleton().uuid, 'session_uuid': session_uuid},
                               timeout_seconds=15,
                               daemon=True)
        self.writer = Writer(api_caller, frequency=computer_singleton().web_api.frequency)
        self.header = Header(api_caller,
                             frequency=computer_singleton().web_api.frequency,
                             open_browser=computer_singleton().web_api.open_browser)
        self.data = {}
        self.cache = {}

    def start(self, configs: Dict[str, any]):
        self.header.start(configs)

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

    def track(self):
        self.track_net_io_counters()
        # inspect(psutil.net_if_addrs())
        # inspect(psutil.net_if_stats())
        self.track_memory()
        self.track_cpu()
        self.track_disk()
        # track_processes()

        self.writer.track(self.data)
        self.data = {}


def get_os():
    if psutil.MACOS:
        return 'macos'
    elif psutil.LINUX:
        return 'linux'
    elif psutil.WINDOWS:
        return 'windows'
    else:
        return 'unknown'

