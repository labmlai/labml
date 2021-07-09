from typing import Tuple, List, Any

from torch import nn
from torch.utils.data import DataLoader

from labml import experiment, tracker, monit
from labml.configs import option, calculate
from labml_helpers.metrics.collector import Collector
from labml_helpers.train_valid import SimpleTrainValidConfigs, BatchIndex
from labml_samples.pytorch.stocks.dataset import MinutelyData, MinutelyDataset
from labml_samples.pytorch.stocks.model import CnnModel


class Configs(SimpleTrainValidConfigs):
    epochs = 32
    dropout: float
    validation_dates: int = 100

    loss_func = nn.MSELoss()

    dataset: MinutelyData
    train_dataset: MinutelyDataset
    valid_dataset: MinutelyDataset
    model: CnnModel

    conv_sizes: List[Tuple[int, int]]
    activation: nn.Module

    train_batch_size: int = 32
    valid_batch_size: int = 64

    output_collector = Collector('output')

    def initialize(self):
        self.train_dataset = self.dataset.train_dataset()
        self.valid_dataset = self.dataset.valid_dataset()
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.train_batch_size,
                                       shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.valid_batch_size,
                                       shuffle=False)
        self.state_modules = [self.output_collector]
        tracker.set_tensor('output.*')

    def step(self, batch: Any, batch_idx: BatchIndex):
        self.model.train(self.mode.is_train)
        data, target = batch['data'].to(self.device), batch['target'].to(self.device)
        target = (target - self.model.y_mean) / self.model.y_std

        if self.mode.is_train:
            tracker.add_global_step(len(data))

        output = self.model(data)
        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        if self.mode.is_train:
            loss.backward()
            if batch_idx.is_last:
                tracker.add('model', self.model)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if not self.mode.is_train:
            self.output_collector(output * self.model.y_std + self.model.y_mean)

        tracker.save()


@option(Configs.dataset)
def dataset(c: Configs):
    return MinutelyData(c.validation_dates)


@option(Configs.model)
def cnn_model(c: Configs):
    return CnnModel(price_mean=c.train_dataset.price_mean,
                    price_std=c.train_dataset.price_std,
                    volume_mean=c.train_dataset.volume_mean,
                    volume_std=c.train_dataset.volume_std,
                    y_mean=c.train_dataset.y_mean,
                    y_std=c.train_dataset.y_std,
                    activation=c.activation,
                    conv_sizes=c.conv_sizes,
                    dropout=c.dropout).to(c.device)


calculate(Configs.activation, 'relu', [], lambda: nn.ReLU())
calculate(Configs.activation, 'sigmoid', [], lambda: nn.Sigmoid())


def main():
    experiment.create()
    conf = Configs()
    conf.activation = 'relu'
    conf.dropout = 0.1
    experiment.configs(conf,
                       {'conv_sizes': [(128, 2), (256, 4)],
                        'optimizer.learning_rate': 1e-4,
                        'optimizer.optimizer': 'Adam'})

    with experiment.start():
        with monit.section('Initialize'):
            conf.initialize()
        with tracker.namespace('valid'):
            conf.valid_dataset.save_artifacts()
        conf.run()


if __name__ == '__main__':
    main()
