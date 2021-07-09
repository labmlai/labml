import datetime

import torch.distributed
from torch.nn.parallel import DistributedDataParallel

from labml import experiment, monit
from labml.configs import option
from labml.logger import inspect
from labml_samples.pytorch.mnist.e_labml_helpers import Configs, Net


@option(Configs.model)
def ddp_model(c: Configs):
    return DistributedDataParallel(Net().to(c.device), device_ids=[c.device])


def main(local_rank, rank, world_size, uuid, init_method: str = 'tcp://localhost:23456'):
    with monit.section('Distributed'):
        torch.distributed.init_process_group("gloo",
                                             timeout=datetime.timedelta(seconds=30),
                                             init_method=init_method,
                                             rank=rank,
                                             world_size=world_size)
    conf = Configs()
    experiment.create(uuid=uuid, name='mnist ddp')
    experiment.distributed(local_rank, world_size)
    experiment.configs(conf,
                       {'optimizer.optimizer': 'Adam',
                        'optimizer.learning_rate': 1e-4,
                        'model': 'ddp_model',
                        'device.cuda_device': local_rank})
    conf.set_seed.set()
    experiment.add_pytorch_models(dict(model=conf.model))
    with experiment.start():
        conf.run()


def _launcher():
    import os
    world_size = int(os.environ['WORLD_SIZE'])
    run_uuid = os.environ['RUN_UUID']
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    inspect(world_size=os.environ['WORLD_SIZE'],
            run_uuid=os.environ['RUN_UUID'],
            local_rank=os.environ['LOCAL_RANK'],
            rank=os.environ['RANK'],
            master_addr=os.environ['MASTER_ADDR'],
            master_port=os.environ['MASTER_PORT'])
    main(local_rank, rank, world_size, run_uuid, 'env://')


if __name__ == '__main__':
    # Run single GPU
    # main(0, 1, experiment.generate_uuid())

    # Spawn multiple GPU
    # torch.multiprocessing.spawn(main, args=(3, experiment.generate_uuid()), nprocs=3, join=True)

    _launcher()
