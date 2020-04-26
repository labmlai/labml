import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

import lab
from lab import tracker, monit, loop, experiment
from lab.helpers import training_loop
from lab.utils import pytorch as pytorch_utils


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


class Configs(training_loop.TrainingLoopConfigs):
    epochs: int = 10

    loop_step = 'loop_step'
    loop_count = 'loop_count'

    is_save_models = True
    batch_size: int = 64
    test_batch_size: int = 1000

    use_cuda: bool = True
    cuda_device: int = 0
    seed: int = 5
    train_log_interval: int = 10

    is_log_parameters: bool = True

    device: any

    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD

    set_seed = 'set_seed'

    def train(self):
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

    def test(self):
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

    def run(self):
        pytorch_utils.add_model_indicators(self.model)

        tracker.set_queue("train_loss", 20, True)
        tracker.set_histogram("test_loss", True)
        tracker.set_histogram("accuracy", True)

        for _ in self.training_loop:
            self.train()
            self.test()
            if self.is_log_parameters:
                pytorch_utils.store_model_indicators(self.model)


@Configs.calc(Configs.device)
def device(c: Configs):
    from lab.utils.pytorch import get_device

    return get_device(c.use_cuda, c.cuda_device)


def _data_loader(is_train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST(str(lab.get_data_path()),
                       train=is_train,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)


@Configs.calc([Configs.train_loader, Configs.test_loader])
def data_loaders(c: Configs):
    train = _data_loader(True, c.batch_size)
    test = _data_loader(False, c.test_batch_size)

    return train, test


@Configs.calc(Configs.model)
def model(c: Configs):
    m: Net = Net()
    m.to(c.device)
    return m


@Configs.calc(Configs.optimizer)
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


@Configs.calc(Configs.loop_step)
def loop_step(c: Configs):
    return len(c.train_loader)


def main():
    conf = Configs()
    experiment.create(writers={'sqlite', 'tensorboard'})
    experiment.calculate_configs(conf,
                            {'optimizer': 'adam_optimizer'},
                            ['set_seed', 'run'])
    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
