import time

from labml import experiment
from labml_remote.job import JOBS
from labml_remote.server import SERVERS

USE_ENV = False

PROC_PER_NODE = 1
N_NODES = len([s for s in SERVERS])
MASTER_PORT = 1234

run_uuid = experiment.generate_uuid()
master_addr = None
world_size = N_NODES * PROC_PER_NODE

for i, server in enumerate(SERVERS):
    if master_addr is None:
        master_addr = SERVERS[server].conf.hostname

    for local_rank in range(PROC_PER_NODE):
        rank = i * PROC_PER_NODE + local_rank
        env_vars = {'GLOO_SOCKET_IFNAME': 'enp1s0',
                    'RUN_UUID': run_uuid,
                    'MASTER_ADDR': master_addr,
                    'MASTER_PORT': f'{MASTER_PORT}',
                    'WORLD_SIZE': f'{world_size}',
                    'RANK': f'{rank}',
                    'LOCAL_RANK': f'{local_rank}'}

        if PROC_PER_NODE > 1:
            env_vars['OMP_NUM_THREADS'] = '1'

        cmd = ['python', 'mnist.py']

        if not USE_ENV:
            cmd += [f'--local_rank={local_rank}']

        tags = ['mnist']
        if i == 0 and local_rank == 0:
            tags += ['master']

        JOBS.create(server, ' '.join(cmd), env_vars, tags).start()
        time.sleep(1)
