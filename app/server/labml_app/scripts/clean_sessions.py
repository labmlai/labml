from labml_app.db import init_db
from labml_app.analyses.computers import process, battery, cpu, disk, gpu, network, memory

init_db()

process_keys = process.ProcessModel.get_all()
for process_key in process_keys:
    p = process_key.load()
    p.delete()

process_keys = battery.BatteryModel.get_all()
for process_key in process_keys:
    p = process_key.load()
    p.delete()

process_keys = cpu.CPUModel.get_all()
for process_key in process_keys:
    p = process_key.load()
    p.delete()

process_keys = disk.DiskModel.get_all()
for process_key in process_keys:
    p = process_key.load()
    p.delete()

process_keys = gpu.GPUModel.get_all()
for process_key in process_keys:
    p = process_key.load()
    p.delete()

process_keys = network.NetworkModel.get_all()
for process_key in process_keys:
    p = process_key.load()
    p.delete()

process_keys = memory.MemoryModel.get_all()
for process_key in process_keys:
    p = process_key.load()
    p.delete()
