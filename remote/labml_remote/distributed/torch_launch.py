import time
from typing import Optional, Dict, List

from labml import experiment
from labml_remote.job import JOBS
from labml_remote.server import SERVERS


def launch(python_cmd: str, *,
           tags: List[str],
           n_proc_per_node: int,
           use_env: bool = False,
           master_port: int = 1234,
           env_vars: Optional[Dict[str, str]] = None):
    n_nodes = len([s for s in SERVERS])

    run_uuid = experiment.generate_uuid()
    master_addr = None
    world_size = n_nodes * n_proc_per_node

    if env_vars is None:
        env_vars = {}

    for node_rank, server in enumerate(SERVERS):
        if master_addr is None:
            master_addr = SERVERS[server].conf.hostname

        for local_rank in range(n_proc_per_node):
            rank = node_rank * n_proc_per_node + local_rank
            proc_env_vars = {'RUN_UUID': run_uuid,
                             'MASTER_ADDR': master_addr,
                             'MASTER_PORT': f'{master_port}',
                             'WORLD_SIZE': f'{world_size}',
                             'NODE_RANK': f'{node_rank}',
                             'RANK': f'{rank}',
                             'LOCAL_RANK': f'{local_rank}'}

            if n_proc_per_node > 1:
                proc_env_vars['OMP_NUM_THREADS'] = '1'

            proc_env_vars.update(env_vars)
            cmd = ['python', python_cmd]

            if not use_env:
                cmd += [f'--local_rank={local_rank}']

            proc_tags = tags.copy()
            if node_rank == 0 and local_rank == 0:
                proc_tags += ['master']

            JOBS.create(server, ' '.join(cmd), proc_env_vars, proc_tags).start()
            time.sleep(1)
