import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from labml_helpers.datasets.mnist import MNISTConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.seed import SeedConfigs
from labml_helpers.train_valid import TrainValidConfigs

from labml import experiment
from labml.configs import option


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleAccuracy:
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> int:
        pred = output.argmax(dim=1)
        return pred.eq(target).sum().item()


class Configs(MNISTConfigs, TrainValidConfigs):
    seed = SeedConfigs()
    device: torch.device = DeviceConfigs()
    epochs: int = 10

    is_save_models = True
    model: nn.Module

    loss_func = nn.CrossEntropyLoss()
    accuracy_func = SimpleAccuracy()


@option(Configs.model)
def model(c: Configs):
    return Net().to(c.device)


def main():
    conf = Configs()
    experiment.create(name='mnist_latest')
    experiment.configs(conf,
                       {'device.cuda_device': 0,
                        'optimizer.optimizer': 'Adam'})
    experiment.add_pytorch_models(dict(model=conf.model))
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
