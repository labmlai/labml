from typing import Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

from lab import logger, configs, IndicatorOptions, IndicatorType
from lab.experiment.pytorch import Experiment

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
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return func.log_softmax(x, dim=1)


class Loop:
    def __init__(self, loop_configs):
        self.__epochs = loop_configs.epochs
        self.__is_save_models = loop_configs.is_save_models
        self.__is_log_parameters = loop_configs.is_log_parameters
        self.__log_new_line_interval = loop_configs.log_new_line_interval
        print(self.__epochs)

    def step(self, epoch):
        raise NotImplementedError()

    def __log_model_params(self):
        if not self.__is_log_parameters:
            return

        # Add histograms with model parameter values and gradients
        for model_name, m in MODELS.items():
            for name, param in m.named_parameters():
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

    def startup(self):
        # Start the experiment
        for model_name, m in MODELS.items():
            for name, param in m.named_parameters():
                if param.requires_grad:
                    logger.add_indicator(f"{model_name}.{name}",
                                         IndicatorType.histogram,
                                         IndicatorOptions(is_print=False))
                    logger.add_indicator(f"{model_name}.{name}.grad",
                                         IndicatorType.histogram,
                                         IndicatorOptions(is_print=False))

    def __call__(self):
        self.startup()
        self.loop()


class MNISTLoop(Loop):
    def __init__(self, c: 'Configs'):
        super().__init__(loop_configs=c.loop_configs)
        self.model = c.model
        self.device = c.device
        self.train_loader = c.train_loader
        self.test_loader = c.test_loader
        self.optimizer = c.optimizer
        self.log_interval = c.log_interval

    def startup(self):
        super().startup()

        logger.add_indicator("train_loss", IndicatorType.queue,
                             IndicatorOptions(queue_size=20, is_print=True))
        logger.add_indicator("test_loss", IndicatorType.histogram,
                             IndicatorOptions(is_print=True))
        logger.add_indicator("accuracy", IndicatorType.histogram,
                             IndicatorOptions(is_print=True))

    def _train(self):
        self.model.train()
        for i, (data, target) in logger.enumerator("Train", self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = func.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            # Add training loss to the logger.
            # The logger will queue the values and output the mean
            logger.store(train_loss=loss.item())
            logger.add_global_step()

            # Print output to the console
            if i % self.log_interval == 0:
                # Output the indicators
                logger.write()
                logger.save_checkpoint()

    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in logger.iterator("Test", self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += func.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Add test loss and accuracy to logger
        logger.store(test_loss=test_loss / len(self.test_loader.dataset))
        logger.store(accuracy=correct / len(self.test_loader.dataset))

    def step(self, epoch):
        # Training and testing
        self._train()
        self._test()


class LoopConfigsTuple(NamedTuple):
    epochs: int = 10
    is_save_models: bool = False
    is_log_parameters: bool = True
    log_new_line_interval: int = 1


class LoopConfigs(configs.Configs):
    epochs: int = 10
    is_save_models: bool = False
    is_log_parameters: bool = True
    log_new_line_interval: int = 1

    loop_configs: LoopConfigsTuple


@LoopConfigs.calc('loop_configs')
def _loop_configs(c: LoopConfigs):
    return LoopConfigsTuple(epochs=c.epochs,
                            is_save_models=c.is_save_models,
                            is_log_parameters=c.is_log_parameters,
                            log_new_line_interval=c.log_new_line_interval)


class LoaderConfigs(configs.Configs):
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader


class Configs(LoopConfigs, LoaderConfigs):
    batch_size: int = 64
    test_batch_size: int = 1000

    # Reset epochs so that it'll be computed
    epochs: int = None
    use_cuda: float = True
    cuda_device: str = "cuda:1"
    seed: int = 5
    log_interval: int = 10

    loop: MNISTLoop

    is_cuda: bool

    device: any

    data_loader_args: Dict

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD

    set_seed = None


@Configs.calc('epochs')
def from_batch(c: Configs):
    return 2 * c.batch_size


@Configs.calc('epochs')
def random(c: Configs):
    return c.seed


# Get dependencies from parameters.
# The code looks cleaner, but might cause problems when you want to refactor
# later.
# It will be harder to use static analysis tools to find the usage of configs.
@Configs.calc(['is_cuda', 'device', 'data_loader_args'])
def cuda(*, use_cuda, cuda_device):
    is_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device(cuda_device if is_cuda else "cpu")
    dl_args = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    return is_cuda, device, dl_args


def _data_loader(is_train, batch_size, data_loader_args):
    return torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=is_train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **data_loader_args)


# The value name is inferred from the function name
@Configs.calc()
def train_loader(c: Configs):
    with logger.section("Training data"):
        return _data_loader(True, c.batch_size, c.data_loader_args)


@Configs.calc()
def test_loader(c: Configs):
    with logger.section("Testing data"):
        return _data_loader(False, c.test_batch_size, c.data_loader_args)


# Compute multiple results from a single function
@Configs.calc(['model', 'optimizer'])
def model_optimizer(c: Configs):
    with logger.section("Create model"):
        m: Net = Net()
        m.to(c.device)

    with logger.section("Create optimizer"):
        o = optim.SGD(m.parameters(), lr=c.learning_rate, momentum=c.momentum)

    return m, o


@Configs.calc()
def set_seed(c: Configs):
    with logger.section("Setting seed"):
        torch.manual_seed(c.seed)


def main():
    conf = Configs()
    experiment = Experiment(writers={'sqlite'})
    experiment.calc_configs(conf,
                            {'epochs': 'random'},
                            ['set_seed', 'loop'])
    experiment.add_models(MODELS)
    experiment.start()
    conf.loop()


if __name__ == '__main__':
    main()
