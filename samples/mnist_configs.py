import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from lab import logger, configs
from lab.experiment.pytorch import Experiment

# Declare the experiment
EXPERIMENT = Experiment(name="mnist_pytorch",
                        python_file=__file__,
                        comment="Test",
                        check_repo_dirty=False
                        )

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
                                         indicator_type='histogram',
                                         is_print=False)
                    logger.add_indicator(f"{model_name}.{name}.grad",
                                         indicator_type='histogram',
                                         is_print=False)

        self.startup()

        self.loop()


class MNISTLoop(Loop):
    def __init__(self, *, model, device, train_loader, test_loader, optimizer,
                 log_interval, epochs, is_save_models, is_log_parameters,
                 log_new_line_interval):
        super().__init__(epochs=epochs,
                         is_save_models=is_save_models,
                         is_log_parameters=is_log_parameters,
                         log_new_line_interval=log_new_line_interval)
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.log_interval = log_interval

    def startup(self):
        logger.add_indicator("train_loss", indicator_type='queue', queue_limit=20)
        logger.add_indicator("test_loss", indicator_type='histogram')
        logger.add_indicator("accuracy", indicator_type='histogram')

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
    learning_rate: float = 0.01
    momentum: float = 0.5
    use_cuda: float = True
    cuda_device: str = "cuda"
    seed: int = 5
    log_interval: int = 10

    def is_cuda(self, *, use_cuda: bool) -> bool:
        return use_cuda and torch.cuda.is_available()

    def device(self, *, is_cuda: bool, cuda_device: str):
        return torch.device(cuda_device if is_cuda else "cpu")

    def data_loader_args(self, *, is_cuda):
        return {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

    def train_loader(self, *, batch_size, data_loader_args):
        with logger.section("Training data"):
            return torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=batch_size, shuffle=True, **data_loader_args)

    def test_loader(self, *, test_batch_size, data_loader_args):
        return torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=True, **data_loader_args)

    def model(self, *, device, set_seed):
        with logger.section("Create model"):
            return Net().to(device)

    def optimizer(self, *, model, learning_rate, momentum):
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    def set_seed(self, *, seed: int):
        torch.manual_seed(seed)

    loop = MNISTLoop


def main():
    conf = Configs()
    proc = configs.ConfigProcessor(conf)
    proc.calculate()
    loop = proc.computed['loop']
    loop()


if __name__ == '__main__':
    main()
