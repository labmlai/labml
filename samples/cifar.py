import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

import lab
from lab import tracker, monit, loop, experiment
from lab.configs import BaseConfigs
from lab.helpers.pytorch.device import DeviceConfigs
from lab.helpers.training_loop import TrainingLoopConfigs
from lab.utils import pytorch as pytorch_utils


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


net = Net()


class CIFAR:
    def __init__(self, c: 'Configs'):
        self.model = c.model
        self.device = c.device
        self.train_loader = c.train_loader
        self.test_loader = c.test_loader
        self.optimizer = c.optimizer
        self.train_log_interval = c.train_log_interval
        self.loop = c.training_loop
        self.__is_log_parameters = c.is_log_parameters

    def _train(self):
        self.model.train()
        for i, (data, target) in monit.enum("Train", self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            tracker.add(train_loss=loss)
            loop.add_global_step()

            if i % self.train_log_interval == 0:
                tracker.save()

    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in monit.iterate("Test", self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        tracker.add(test_loss=test_loss / len(self.test_loader.dataset))
        tracker.add(accuracy=correct / len(self.test_loader.dataset))

    def __log_model_params(self):
        if not self.__is_log_parameters:
            return

        pytorch_utils.store_model_indicators(self.model)

    def __call__(self):
        pytorch_utils.add_model_indicators(self.model)

        tracker.set_queue("train_loss", 20, True)
        tracker.set_histogram("test_loss", True)
        tracker.set_histogram("accuracy", True)

        for _ in self.loop:
            self._train()
            self._test()
            self.__log_model_params()


class LoaderConfigs(BaseConfigs):
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader


class Configs(DeviceConfigs, TrainingLoopConfigs, LoaderConfigs):
    epochs: int = 10

    transforms: any

    loop_step = 'loop_step'
    loop_count = 'loop_count'

    is_save_models = True
    batch_size: int = 64
    test_batch_size: int = 1000

    seed: int = 5
    train_log_interval: int = 10

    is_log_parameters: bool = True

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD

    set_seed = 'set_seed'

    main: CIFAR


@Configs.calc(Configs.transforms)
def cifar_transforms():
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _data_loader(is_train, batch_size, trans):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(str(lab.get_data_path()),
                         train=is_train,
                         download=True,
                         transform=trans),
        batch_size=batch_size, shuffle=True)


@Configs.calc([Configs.train_loader, Configs.test_loader])
def data_loaders(c: Configs):
    train = _data_loader(True, c.batch_size, c.transforms)
    test = _data_loader(False, c.test_batch_size, c.transforms)

    return train, test


@Configs.calc(Configs.model)
def model(c: Configs):
    m: Net = Net()
    m.to(c.device)
    return m


@Configs.calc('optimizer')
def sgd_optimizer(c: Configs):
    return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)


@Configs.calc(Configs.optimizer)
def adam_optimizer(c: Configs):
    return optim.Adam(c.model.parameters(), lr=c.learning_rate)


@Configs.calc(Configs.set_seed)
def set_seed(c: Configs):
    torch.manual_seed(c.seed)


@Configs.calc(Configs.loop_count)
def loop_count(c: Configs):
    return c.epochs * len(c.train_loader)


@Configs.calc()
def loop_step(c: Configs):
    return len(c.train_loader)


def main():
    conf = Configs()
    experiment.create(writers={'sqlite'})
    experiment.calculate_configs(conf,
                                 {'optimizer': 'adam_optimizer'},
                                 ['set_seed', 'main'])
    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()
    conf.main()


if __name__ == '__main__':
    main()
