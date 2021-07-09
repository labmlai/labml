"""A sample that uses the API to launch a set of processes"""

import time

from labml import experiment
from labml_remote.job import JOBS
from labml_remote.server import SERVERS

PROC_PER_NODE = 1
TAGS = ['mnist']


def start_jobs():
    n_nodes = len([s for s in SERVERS])
    run_uuid = experiment.generate_uuid()
    master_addr = None
    world_size = n_nodes * PROC_PER_NODE

    for node_rank, server in enumerate(SERVERS):
        if master_addr is None:
            master_addr = SERVERS[server].conf.hostname

        for local_rank in range(PROC_PER_NODE):
            rank = node_rank * PROC_PER_NODE + local_rank
            env_vars = {'GLOO_SOCKET_IFNAME': 'enp1s0',
                        'RUN_UUID': run_uuid,
                        'MASTER_ADDR': master_addr,
                        'MASTER_PORT': f'{1234}',
                        'WORLD_SIZE': f'{world_size}',
                        'RANK': f'{rank}',
                        'LOCAL_RANK': f'{local_rank}'}

            if PROC_PER_NODE > 1:
                env_vars['OMP_NUM_THREADS'] = '1'

            cmd = ['python', 'mnist.py']

            tags = TAGS.copy()
            if node_rank == 0 and local_rank == 0:
                tags += ['master']

            JOBS.create(server, ' '.join(cmd), env_vars, tags).start()
            time.sleep(1)


if __name__ == '__main__':
    start_jobs()
