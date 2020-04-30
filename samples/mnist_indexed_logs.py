import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from lab import tracker, monit, loop, experiment
from lab.helpers.pytorch.datasets.mnist import MNISTConfigs
from lab.helpers.pytorch.device import DeviceConfigs
from lab.helpers.training_loop import TrainingLoopConfigs
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
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Configs(MNISTConfigs, DeviceConfigs, TrainingLoopConfigs):
    epochs: int = 10

    loop_step = 'loop_step'
    loop_count = 'loop_count'

    is_save_models = True

    # Reset epochs so that it'll be computed
    seed: int = 5
    train_log_interval: int = 10

    is_log_parameters: bool = True

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
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            # Add training loss to the logger.
            # The logger will queue the values and output the mean
            tracker.add(train_loss=loss)
            loop.add_global_step()

            # Print output to the console
            if i % self.train_log_interval == 0:
                # Output the indicators
                tracker.save()

    def valid(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        idx = 0
        with torch.no_grad():
            for data, target in monit.iterate("Test", self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.nll_loss(output, target, reduction='none')
                values = list(loss.cpu().numpy())
                indexes = [idx + i for i in range(len(values))]
                tracker.add('test_sample_loss', (indexes, values))

                test_loss += float(np.sum(loss.cpu().numpy()))
                pred = output.argmax(dim=1, keepdim=True)
                values = list(pred.cpu().numpy())
                tracker.add('test_sample_pred', (indexes, values))
                correct += pred.eq(target.view_as(pred)).sum().item()

                idx += len(values)

        # Add test loss and accuracy to logger
        tracker.add(test_loss=test_loss / len(self.valid_dataset))
        tracker.add(accuracy=correct / len(self.valid_dataset))

    def run(self):
        # Training and testing
        pytorch_utils.add_model_indicators(self.model)

        tracker.set_queue("train_loss", 20, True)
        tracker.set_histogram("test_loss", True)
        tracker.set_histogram("accuracy", True)
        tracker.set_indexed_scalar('test_sample_loss')
        tracker.set_indexed_scalar('test_sample_pred')

        test_data = np.array([d[0].numpy() for d in self.valid_dataset])
        experiment.save_numpy("test_data", test_data)

        for _ in self.training_loop:
            self.train()
            self.valid()
            if self.is_log_parameters:
                pytorch_utils.store_model_indicators(self.model)


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
    experiment.create(writers={'sqlite'})
    conf.optimizer = 'adam_optimizer'
    experiment.calculate_configs(conf,
                                 {},
                                 ['set_seed', 'run'])
    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
