import datetime

import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from labml import experiment, monit
from labml import tracker
from labml.configs import option
from labml.logger import inspect
from labml.utils import pytorch as pytorch_utils
from labml_helpers.datasets.mnist import MNISTConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.module import Module
from labml_helpers.seed import SeedConfigs
from labml_helpers.train_valid import TrainValidConfigs, BatchIndex
from torch.nn.parallel import DistributedDataParallel


class Net(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def __call__(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Configs(MNISTConfigs, TrainValidConfigs):
    optimizer: torch.optim.Adam
    model: nn.Module
    set_seed = SeedConfigs()
    device: torch.device = DeviceConfigs()
    epochs: int = 10

    is_save_models = True
    model: nn.Module
    inner_iterations = 10

    accuracy_func = Accuracy()
    loss_func = nn.CrossEntropyLoss()

    def init(self):
        tracker.set_queue("loss.*", 20, True)
        tracker.set_scalar("accuracy.*", True)
        self.state_modules = [self.accuracy_func]

    def step(self, batch: any, batch_idx: BatchIndex):
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(len(data))

        with self.mode.update(is_log_activations=batch_idx.is_last):
            output = self.model(data)

        loss = self.loss_func(output, target)
        self.accuracy_func(output, target)
        tracker.add("loss.", loss)

        if self.mode.is_train:
            loss.backward()

            self.optimizer.step()
            if batch_idx.is_last:
                pytorch_utils.store_model_indicators(self.model)
            self.optimizer.zero_grad()

        tracker.save()


@option(Configs.model)
def model(c: Configs):
    return Net().to(c.device)


@option(Configs.optimizer)
def _optimizer(c: Configs):
    from labml_helpers.optimizer import OptimizerConfigs
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.model.parameters()
    return opt_conf


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
