import torch

from labml.configs import BaseConfigs, option, meta_config


class OptimizerConfigs(BaseConfigs):
    optimizer: torch.optim.Adam
    learning_rate: float = 0.01
    momentum: float = 0.5
    parameters: any
    d_model: int = None

    def __init__(self):
        super().__init__(_primary='optimizer')


meta_config(OptimizerConfigs.parameters)


@option(OptimizerConfigs.optimizer, 'SGD')
def sgd_optimizer(c: OptimizerConfigs):
    return torch.optim.SGD(c.parameters, c.learning_rate, c.momentum)


@option(OptimizerConfigs.optimizer, 'Adam')
def adam_optimizer(c: OptimizerConfigs):
    return torch.optim.Adam(c.parameters, c.learning_rate)


class NoamOpt:
    def __init__(self, model_size: int, factor: float, warmup: int, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


@option(OptimizerConfigs.optimizer, 'Noam')
def noam_optimizer(c: OptimizerConfigs):
    optimizer = torch.optim.Adam(c.parameters, lr=c.learning_rate)
    return NoamOpt(c.d_model, 1, 2000, optimizer)


def _test_noam_optimizer():
    import matplotlib.pyplot as plt
    import numpy as np

    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.title("Optimizer")
    plt.show()


if __name__ == '__main__':
    _test_noam_optimizer()
