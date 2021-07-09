r"""
This is an example of how you can inherit configs.
"""

import torch.nn as nn
import torch.nn.functional as F

from labml import experiment
from labml.configs import option
from labml_helpers.datasets.cifar10 import CIFAR10Configs
from labml_samples.pytorch.mnist.e_labml_helpers import Configs as MNISTExperimentConfigs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Configs(CIFAR10Configs, MNISTExperimentConfigs):
    model: nn.Module = 'cifar10_model'
    dataset_name = 'CIFAR10'


@option(Configs.model)
def cifar10_model(c: Configs):
    return Net().to(c.device)


def main():
    conf = Configs()
    experiment.create(name='cifar_10')
    experiment.configs(conf,
                       {'optimizer.optimizer': 'Adam',
                        'optimizer.learning_rate': 1e-4})
    conf.set_seed.set()
    experiment.add_pytorch_models(dict(model=conf.model))
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
