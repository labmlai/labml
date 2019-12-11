from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from lab import logger, configs, IndicatorOptions, IndicatorType
from lab.experiment.pytorch import Experiment

# Declare the experiment
EXPERIMENT = Experiment(writers={'sqlite'})

MODELS = {}


class Model(nn.Module):
    """Can intercept all the model calls"""

    def __init__(self, name):
        super().__init__()
        MODELS[name] = self


class Net(Model):
    def __init__(self):
        super().__init__('MyModel')
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


class Loop:
    def __init__(self, *, epochs, is_save_models, is_log_parameters,
                 log_new_line_interval):
        self.__epochs = epochs
        self.__is_save_models = is_save_models
        self.__is_log_parameters = is_log_parameters
        self.__log_new_line_interval = log_new_line_interval

    def startup(self):
        pass

    def step(self, epoch):
        raise NotImplementedError()

    def __log_model_params(self):
        if not self.__is_log_parameters:
            return

        # Add histograms with model parameter values and gradients
        for model_name, model in MODELS.items():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    logger.store(f"{model_name}.{name}", param.data.cpu().numpy())
                    logger.store(f"{model_name}.{name}.grad", param.grad.cpu().numpy())

    def loop(self):
        # Loop through the monitored iterator
        for epoch in logger.loop(range(0, self.__epochs)):
            # Delayed keyboard interrupt handling to use
            # keyboard interrupts to end the loop.
            # This will capture interrupts and finish
            # the loop at the end of processing the iteration;
            # i.e. the loop won't stop in the middle of an epoch.
            try:
                with logger.delayed_keyboard_interrupt():
                    self.step(epoch)

                    self.__log_model_params()

                    # Clear line and output to console
                    logger.write()

                    # Clear line and go to the next line;
                    # that is, we add a new line to the output
                    # at the end of each epoch
                    if (epoch + 1) % self.__log_new_line_interval == 0:
                        logger.new_line()

            # Handled delayed interrupt
            except KeyboardInterrupt:
                logger.finish_loop()
                logger.new_line()
                logger.log("\nKilling loop...")
                break

    def __call__(self):
        # Start the experiment
        for model_name, model in MODELS.items():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    logger.add_indicator(f"{model_name}.{name}",
                                         IndicatorType.histogram,
                                         IndicatorOptions(is_print=False))
                    logger.add_indicator(f"{model_name}.{name}.grad",
                                         IndicatorType.histogram,
                                         IndicatorOptions(is_print=False))

        self.startup()

        self.loop()


class MNISTLoop(Loop):
    def __init__(self, c: 'Configs'):
        super().__init__(epochs=c.epochs,
                         is_save_models=c.is_save_models,
                         is_log_parameters=c.is_log_parameters,
                         log_new_line_interval=c.log_new_line_interval)
        self.model = c.model
        self.device = c.device
        self.train_loader = c.train_loader
        self.test_loader = c.test_loader
        self.optimizer = c.optimizer
        self.log_interval = c.log_interval

    def startup(self):
        logger.add_indicator("train_loss", IndicatorType.queue,
                             IndicatorOptions(queue_size=20, is_print=True))
        logger.add_indicator("test_loss", IndicatorType.histogram,
                             IndicatorOptions(is_print=True))
        logger.add_indicator("accuracy", IndicatorType.histogram,
                             IndicatorOptions(is_print=True))

        EXPERIMENT.start_train()

    def _train(self, epoch):
        with logger.section("Train", total_steps=len(self.train_loader)):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                # Add training loss to the logger.
                # The logger will queue the values and output the mean
                logger.store(train_loss=loss.item())
                logger.progress(batch_idx + 1)
                logger.set_global_step(epoch * len(self.train_loader) + batch_idx)

                # Print output to the console
                if batch_idx % self.log_interval == 0:
                    # Output the indicators
                    logger.write()

    def _test(self):
        with logger.section("Test", total_steps=len(self.test_loader)):
            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    logger.progress(batch_idx + 1)

            # Add test loss and accuracy to logger
            logger.store(test_loss=test_loss / len(self.test_loader.dataset))
            logger.store(accuracy=correct / len(self.test_loader.dataset))

    def step(self, epoch):
        # Training and testing
        self._train(epoch)
        self._test()


class BaseConfigs(configs.Configs):
    is_save_models: bool = False
    is_log_parameters: bool = True
    log_new_line_interval: int = 1


class Configs(BaseConfigs):
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 10
    use_cuda: float = True
    cuda_device: str = "cuda:1"
    seed: int = 5
    log_interval: int = 10

    loop: MNISTLoop

    is_cuda: bool

    device: any

    data_loader_args: Dict

    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD


@Configs.calc()
def is_cuda(c: Configs) -> bool:
    return c.use_cuda and torch.cuda.is_available()


@Configs.calc()
def device(c: Configs):
    return torch.device(c.cuda_device if c.is_cuda else "cpu")


@Configs.calc()
def data_loader_args(c: Configs):
    return {'num_workers': 1, 'pin_memory': True} if c.is_cuda else {}


@Configs.calc()
def train_loader(c: Configs):
    with logger.section("Training data"):
        return torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=c.batch_size, shuffle=True, **c.data_loader_args)


@Configs.calc()
def test_loader(c: Configs):
    return torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=c.test_batch_size, shuffle=True, **c.data_loader_args)


@Configs.calc()
def model(c: Configs):
    with logger.section("Create model"):
        return Net().to(c.device)


@Configs.calc()
def optimizer(c: Configs):
    return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)


@Configs.calc()
def set_seed(c: Configs):
    torch.manual_seed(c.seed)


def main():
    conf = Configs()
    proc = configs.ConfigProcessor(conf)
    proc.calculate()

    conf.loop()


if __name__ == '__main__':
    main()
