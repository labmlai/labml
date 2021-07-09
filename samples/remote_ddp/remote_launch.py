"""
labml_remote setup
labml_remote rsync
labml_remote update

Test:

source ~/miniconda/etc/profile.d/conda.sh
conda activate labml_sample_env
cd labml_sample/
PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src" cmd

export GLOO_SOCKET_IFNAME=enp1s0
export NCCL_SOCKET_IFNAME=enp1s0

RUN_UUID=fba9eb202cb211eba5beacde48001122 PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src" python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=104.171.200.181 --master_port=1234 labml_samples/pytorch/ddp/mnist.py
RUN_UUID=fba9eb202cb211eba5beacde48001122 PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src" python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=104.171.200.181 --master_port=1234 labml_samples/pytorch/ddp/mnist.py

RUN_UUID=fba9eb202cb211eba5beacde48001122 PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src" python -m torch.distributed.launch --nproc_per_node=2 labml_samples/pytorch/ddp/mnist.py
"""

import time

from labml import experiment
from labml_remote.job import JOBS
from labml_remote.server import SERVERS

PROC_PER_NODE = 1
N_NODES = len([s for s in SERVERS])

run_uuid = experiment.generate_uuid()
master_addr = None

for i, server in enumerate(SERVERS):
    if master_addr is None:
        master_addr = SERVERS[server].conf.hostname

    cmd = f'python -m torch.distributed.launch ' \
          f'--nproc_per_node={PROC_PER_NODE} ' \
          f'--nnodes={N_NODES} ' \
          f'--node_rank={i} ' \
          f'--master_addr={master_addr} --master_port=1234 ' \
          f'mnist.py'

    env_vars = {'GLOO_SOCKET_IFNAME': 'enp1s0',
                'RUN_UUID': run_uuid}
    tags = ['mnist']
    if i == 0:
        tags += ['master']
    JOBS.create(server, cmd, env_vars, tags).start()
    time.sleep(1)
