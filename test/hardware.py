from labml import lab, logger
from labml.logger import inspect


def test_nvidia_device(idx: int):
    from py3nvml import py3nvml as nvml

    handle = nvml.nvmlDeviceGetHandleByIndex(idx)

    pciInfo = nvml.nvmlDeviceGetPciInfo(handle)

    brands = {nvml.NVML_BRAND_UNKNOWN: "Unknown",
              nvml.NVML_BRAND_QUADRO: "Quadro",
              nvml.NVML_BRAND_TESLA: "Tesla",
              nvml.NVML_BRAND_NVS: "NVS",
              nvml.NVML_BRAND_GRID: "Grid",
              nvml.NVML_BRAND_GEFORCE: "GeForce"}

    inspect(idx=idx,
            # id=pciInfo.busId,
            # uuid=nvml.nvmlDeviceGetUUID(handle),
            name=nvml.nvmlDeviceGetName(handle),
            # brand=brands[nvml.nvmlDeviceGetBrand(handle)],
            # multi_gpu=nvml.nvmlDeviceGetMultiGpuBoard(handle),
            # pcie_link=nvml.nvmlDeviceGetCurrPcieLinkWidth(handle),

            fan=nvml.nvmlDeviceGetFanSpeed(handle),
            # power=nvml.nvmlDeviceGetPowerState(handle),

            mem_total=nvml.nvmlDeviceGetMemoryInfo(handle).total,
            mem_used=nvml.nvmlDeviceGetMemoryInfo(handle).used,

            util_gpu=nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            # util_mem=nvml.nvmlDeviceGetUtilizationRates(handle).memory,

            temp=nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),

            power=nvml.nvmlDeviceGetPowerUsage(handle),
            power_limit=nvml.nvmlDeviceGetPowerManagementLimit(handle),

            # display=nvml.nvmlDeviceGetDisplayMode(handle),
            display_active=nvml.nvmlDeviceGetDisplayActive(handle),
            )

    logger.log()

    procs = nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
    for p in procs:
        inspect(name=nvml.nvmlSystemGetProcessName(p.pid),
                pid=p.pid,
                mem=p.usedGpuMemory)

    procs = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        inspect(name=nvml.nvmlSystemGetProcessName(p.pid),
                pid=p.pid,
                mem=p.usedGpuMemory)

    logger.log()


def test_nvidia():
    # pip install py3nvml
    import py3nvml
    from py3nvml import py3nvml as nvml

    inspect(py3nvml.get_free_gpus())

    nvml.nvmlInit()
    inspect(version=nvml.nvmlSystemGetDriverVersion())
    inspect(count=nvml.nvmlDeviceGetCount())

    for i in range(nvml.nvmlDeviceGetCount()):
        test_nvidia_device(i)

    nvml.nvmlShutdown()


def test_psutil():
    # sudo apt-get install gcc python3-dev
    # xcode on mac
    # pip install psutil
    import psutil

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
    inspect(psutil.Process().as_dict())
    inspect([p for p in psutil.process_iter()])
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


def test_psutil_processes():
    import psutil

    for p in psutil.process_iter():
        inspect(p)


if __name__ == '__main__':
    test_nvidia()
    # test_psutil()
    # test_psutil_processes()
