import psutil

from labml import lab
from labml.logger import inspect


def test_psutil():
    # https://psutil.readthedocs.io/en/latest/#
    inspect(mac=psutil.MACOS, linux=psutil.LINUX, windows=psutil.WINDOWS)
    inspect(psutil.net_io_counters()._asdict())
    inspect(psutil.net_if_addrs())
    inspect(psutil.net_if_stats())
    inspect(psutil.virtual_memory()._asdict())
    inspect(psutil.cpu_count())
    inspect(psutil.cpu_times()._asdict())
    inspect(psutil.cpu_stats()._asdict())
    inspect(psutil.cpu_freq()._asdict())
    inspect(psutil.cpu_percent(percpu=True))
    inspect(psutil.disk_usage(lab.get_path())._asdict())
    inspect([p for p in psutil.process_iter()])
    inspect(psutil.Process().as_dict())
    # inspect(psutil.Process().terminate())
    # inspect('test')
    p = psutil.Process()
    with p.oneshot():
        inspect(p.memory_info()._asdict())
        inspect(p.memory_percent())
        inspect(p.cpu_percent(1))
        inspect(p.num_threads())
        inspect(p.threads())
        try:
            inspect(p.cpu_num())
        except AttributeError as e:
            pass
    try:
        inspect(psutil.sensors_temperatures())
    except AttributeError as e:
        pass
    try:
        inspect(psutil.sensors_fans())
    except AttributeError as e:
        pass
    try:
        inspect(psutil.sensors_battery()._asdict())
    except AttributeError as e:
        pass


if __name__ == '__main__':
    test_psutil()
