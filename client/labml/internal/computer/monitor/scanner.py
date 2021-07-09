import time
from pathlib import Path

import psutil

from labml import logger
from labml.internal.computer.monitor.process import ProcessMonitor
from labml.logger import Text
from labml.utils.notice import labml_notice


class Scanner:
    def __init__(self):
        self.data = {}
        self.cache = {}
        self.nvml = None
        self.n_gpu = 0

        try:
            from py3nvml import py3nvml as nvml
            self.nvml = nvml
        except ImportError:
            labml_notice('Install py3nvml to monitor GPUs:\n pip install py3nvml',
                         is_warn=False)

        if self.nvml:
            try:
                self.nvml.nvmlInit()
                self.nvml.nvmlShutdown()
            except self.nvml.NVMLError:
                logger.log('NVML Library not found', Text.warning)
                self.nvml = None

        self.process_monitor = ProcessMonitor(self.nvml)

    def configs(self):
        configs = {
            'os': self.get_os(),
            'cpu.logical': psutil.cpu_count(),
            'cpu.physical': psutil.cpu_count(logical=False),
        }

        configs.update(self._gpu_header())
        configs.update(self._sensor_header())

        return configs

    def _sensor_header(self):
        try:
            sensors = psutil.sensors_temperatures()
        except AttributeError as e:
            return {}

        data = {}

        for k, temps in sensors.items():
            for i, t in enumerate(temps):
                assert isinstance(t, psutil._common.shwtemp)
                data[f'sensor.temp.name.{k}.{i}'] = t.label

        return data

    def _gpu_header(self):
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

        self.data.update(self.process_monitor.track_gpus())

        self.nvml.nvmlShutdown()

    def first_gpu(self):
        if not self.nvml:
            return

        self.nvml.nvmlInit()
        for i in range(self.n_gpu):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
            self.data[f'gpu.memory.total.{i}'] = self.nvml.nvmlDeviceGetMemoryInfo(handle).total
            try:
                self.data[f'gpu.power.limit.{i}'] = self.nvml.nvmlDeviceGetPowerManagementLimit(handle)
            except self.nvml.NVMLError:
                pass

        self.nvml.nvmlShutdown()

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

    def track_sensors(self):
        try:
            sensors = psutil.sensors_temperatures()
        except AttributeError as e:
            return

        for k, temps in sensors.items():
            for i, t in enumerate(temps):
                self.data[f'sensor.temp.{k}.{i}'] = t.current

    def track_processes(self):
        self.data.update(self.process_monitor.track())

    def track_battery(self):
        try:
            battery = psutil.sensors_battery()._asdict()
        except AttributeError as e:
            return
        except FileNotFoundError as e:
            return

        self.data.update({
            'battery.percent': battery['percent'],
            'battery.power_plugged': battery['power_plugged'],
            'battery.secsleft': battery['secsleft'],
        })

    def track(self):
        self.data = {}
        try:
            self.track_net_io_counters()
            self.track_memory()
            self.track_cpu()
            self.track_disk()
            self.track_sensors()
            self.track_battery()
            self.track_processes()
            self.track_gpu()
        except Exception as e:
            print(e)

        return self.data

    def first(self):
        self.data = {}
        self.first_gpu()

        return self.data

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


def _test():
    scanner = Scanner()
    from labml.logger import inspect
    inspect(scanner.configs())
    inspect(scanner.first())
    inspect(scanner.track())


if __name__ == '__main__':
    _test()
