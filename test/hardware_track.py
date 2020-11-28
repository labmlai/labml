import time

import psutil

from labml import lab, experiment, tracker
from labml.logger import inspect


def track_net_io_counters():
    res = psutil.net_io_counters()
    tracker.add({
        'net.recv': res.bytes_recv,
        'net.sent': res.bytes_sent
    })


def track_memory():
    res = psutil.virtual_memory()
    tracker.add({
        'memory.total': res.total,
        'memory.used': res.used,
        'memory.available': res.available,
    })


def track_cpu():
    res = psutil.cpu_times()
    tracker.add({
        'cpu.idle': res.idle,
        'cpu.system': res.system,
        'cpu.user': res.user,
    })
    res = psutil.cpu_freq()
    tracker.add({
        'cpu.freq': res.current,
        'cpu.freq.min': res.min,
        'cpu.freq.max': res.max,
    })
    res = psutil.cpu_percent(percpu=True)
    tracker.add({f'cpu.perc.{i}': p for i, p in enumerate(res)})


def track_disk():
    res = psutil.disk_usage(lab.get_path())
    tracker.add({
        'disk.free': res.free,
        'disk.total': res.total,
        'disk.used': res.used,
    })


def track():
    tracker.set_global_step(int(time.time()))
    track_net_io_counters()
    # inspect(psutil.net_if_addrs())
    # inspect(psutil.net_if_stats())
    track_memory()
    track_cpu()
    track_disk()
    # track_processes()

    tracker.save()


def get_os():
    if psutil.MACOS:
        return 'macos'
    elif psutil.LINUX:
        return 'linux'
    elif psutil.WINDOWS:
        return 'windows'
    else:
        return 'unknown'


def track_process(p: psutil.Process):
    pid = p.pid

    with p.oneshot():
        try:
            res = p.memory_info()
            tracker.add({
                f'process.{p.pid}.{p.name()}.rss': res.rss,
                f'process.{p.pid}.{p.name()}.vms': res.vms
            })
        except psutil.AccessDenied:
            pass
        try:
            res = p.memory_percent()
            tracker.add({
                f'process.{p.pid}.{p.name()}.mem': res,
            })
        except psutil.AccessDenied:
            pass

        try:
            res = p.cpu_percent()
            tracker.add({
                f'process.{p.pid}.{p.name()}.cpu': res,
            })
        except psutil.AccessDenied:
            pass

        try:
            res = p.num_threads()
            tracker.add({
                f'process.{p.pid}.{p.name()}.threads': res,
            })
        except psutil.AccessDenied:
            pass


def track_processes():
    for p in psutil.process_iter():
        track_process(p)


def main():
    experiment.create(name='hardware', writers={'web_api'})
    experiment.configs({
        'os': get_os(),
        'cpu.logical': psutil.cpu_count(),
        'cpu.physical': psutil.cpu_count(logical=False)
    })

    experiment.start()

    # tracker.set_global_step(int(time.time()))
    # res = psutil.cpu_freq()
    # tracker.add({
    #     'cpu.freq.min': res['min'],
    #     'cpu.freq.max': res['max'],
    # })

    while True:
        track()
        time.sleep(5)


def test_psutil_processes():
    for p in psutil.process_iter():
        inspect(p)
        with p.oneshot():
            try:
                inspect(p.memory_info()._asdict())
                inspect(p.memory_percent())
                inspect(p.cpu_percent(1))
                inspect(p.num_threads())
                inspect(p.threads())
            except psutil.AccessDenied:
                pass

            try:
                inspect(p.cpu_num())
            except AttributeError as e:
                pass


def test_sensors():
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
    # test_psutil_processes()
    # test_sensors()
    main()
