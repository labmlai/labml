import torch

from labml.configs import BaseConfigs, option, meta_config


class OptimizerConfigs(BaseConfigs):
    optimizer: torch.optim.Adam
    learning_rate: float = 0.01
    momentum: float = 0.5
    parameters: any

    def __init__(self):
        super().__init__(_primary='optimizer')


meta_config(OptimizerConfigs.parameters)


@option(OptimizerConfigs.optimizer, 'SGD')
def sgd_optimizer(c: OptimizerConfigs):
    return torch.optim.SGD(c.parameters, c.learning_rate, c.momentum)


@option(OptimizerConfigs.optimizer, 'Adam')
def adam_optimizer(c: OptimizerConfigs):
    return torch.optim.Adam(c.parameters, c.learning_rate)
