from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

from lab import logger, configs
from lab.experiment.pytorch import Experiment
from lab.logger.indicators import Queue, Histogram
from lab.logger.util import pytorch as logger_util


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
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNISTLoop:
    def __init__(self, c: 'Configs'):
        self.model = c.model
        self.device = c.device
        self.train_loader = c.train_loader
        self.test_loader = c.test_loader
        self.optimizer = c.optimizer
        self.log_interval = c.log_interval
        self.__epochs = c.epochs
        self.__is_save_models = c.is_save_models
        self.__is_log_parameters = c.is_log_parameters
        self.__log_new_line_interval = c.log_new_line_interval

    def startup(self):
        logger_util.add_model_indicators(self.model)

        logger.add_indicator(Queue("train_loss", 20, True))
        logger.add_indicator(Histogram("test_loss", True))
        logger.add_indicator(Histogram("accuracy", True))

    def _train(self):
        self.model.train()
        for i, (data, target) in logger.enum("Train", self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            # Add training loss to the logger.
            # The logger will queue the values and output the mean
            logger.store(train_loss=loss)
            logger.add_global_step()

            # Print output to the console
            if i % self.log_interval == 0:
                # Output the indicators
                logger.write()

    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in logger.iterate("Test", self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Add test loss and accuracy to logger
        logger.store(test_loss=test_loss / len(self.test_loader.dataset))
        logger.store(accuracy=correct / len(self.test_loader.dataset))

    def __log_model_params(self):
        if not self.__is_log_parameters:
            return

        # Add histograms with model parameter values and gradients
        logger_util.store_model_indicators(self.model)

    def loop(self):
        # Loop through the monitored iterator
        for epoch in logger.loop(range(0, self.__epochs)):
            self._train()
            self._test()

            self.__log_model_params()

            # Clear line and output to console
            logger.write()

            # Clear line and go to the next line;
            # that is, we add a new line to the output
            # at the end of each epoch
            if (epoch + 1) % self.__log_new_line_interval == 0:
                logger.new_line()

            if self.__is_save_models:
                logger.save_checkpoint()

    def __call__(self):
        self.startup()
        self.loop()


class LoopConfigs(configs.Configs):
    epochs: int = 10
    is_save_models: bool = False
    is_log_parameters: bool = True
    log_new_line_interval: int = 1


class LoaderConfigs(configs.Configs):
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader


class Configs(LoopConfigs, LoaderConfigs):
    batch_size: int = 64
    test_batch_size: int = 1000

    # Reset epochs so that it'll be computed
    epochs: int = 10
    use_cuda: bool = True
    cuda_device: int = 0
    seed: int = 5
    log_interval: int = 10

    loop: MNISTLoop

    device: torch.device

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD

    set_seed = None

    not_used: bool = 10


# Get dependencies from parameters.
# The code looks cleaner, but might cause problems when you want to refactor
# later.
# It will be harder to use static analysis tools to find the usage of configs.
@Configs.calc()
def device(*, use_cuda, cuda_device):
    from lab.util.pytorch import get_device

    return get_device(use_cuda, cuda_device)


def _data_loader(is_train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST(str(logger.get_data_path()),
                       train=is_train,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)


# Multiple configs can be computed from a single function
@Configs.calc(['train_loader', 'test_loader'])
def data_loaders(c: Configs):
    train = _data_loader(True, c.batch_size)

    test = _data_loader(False, c.test_batch_size)

    return train, test


# Compute multiple results from a single function
@Configs.calc()
def model(c: Configs):
    m: Net = Net()
    m.to(c.device)
    return m


# Multiple options for configs can be provided. Option name is inferred from function name,
# unless explicitly provided
@Configs.calc('optimizer')
def sgd_optimizer(c: Configs):
    return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)


@Configs.calc('optimizer')
def adam_optimizer(c: Configs):
    return optim.Adam(c.model.parameters(), lr=c.learning_rate)


# Returns nothing
@Configs.calc()
def set_seed(c: Configs):
    torch.manual_seed(c.seed)


def main():
    conf = Configs()
    experiment = Experiment(writers={'sqlite'})
    experiment.calc_configs(conf,
                            {'optimizer': 'sgd_optimizer'},
                            ['set_seed', 'loop'])
    experiment.add_models(dict(model=conf.model))
    experiment.start()
    conf.loop()


if __name__ == '__main__':
    main()
