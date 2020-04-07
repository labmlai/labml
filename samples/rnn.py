"""
Kaggle Competition https://www.kaggle.com/c/liverpool-ion-switching/overview
download the data from the above line provided
"""

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data

from sklearn.metrics import f1_score

from lab import logger, configs
from lab import training_loop
from lab.experiment.pytorch import Experiment

from lab.util.data.pytorch import CsvDataset


class DataUtils:
    def __init__(self, c: 'Configs'):
        self.device = c.device

    @staticmethod
    def pairs(inputs: list, outputs: list):
        return [(input, output) for input, output in zip(inputs, outputs)]

    def _to_tensor(self, input: float):
        return torch.tensor(input, dtype=torch.long, device=self.device).view(-1, 1)

    def tensors_from_pair(self, pair: tuple):
        input_tensor = self._to_tensor(pair[0])
        target_tensor = self._to_tensor(pair[1])

        return input_tensor, target_tensor


class EncoderRNN(nn.Module):
    def __init__(self, embed_size=8, input_size=1, hidden_size=32):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.layer1 = nn.Linear(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden):
        layer1 = self.layer1(input.view(1, -1).unsqueeze(1))
        output, hidden = self.gru(layer1, hidden)

        output = self.out(output).view(-1)

        return output, hidden

    def init_hidden(self, device):
        return torch.zeros(self.input_size, self.input_size, self.hidden_size, device=device)


class RNN:
    def __init__(self, c: 'Configs'):
        self.batch_size = c.batch_size
        self.epochs = c.epochs

        self.device = c.device

        self.encoder = c.encoder
        self.optimizer = c.optimizer
        self.loss = c.loss

        self.train_loader = c.train_loader
        self.test_loader = c.test_loader

        self.train_log_interval = c.train_log_interval
        self.loop = c.training_loop
        self.__is_log_parameters = c.is_log_parameters

    def _train(self):
        for i, (input_tensors, target_tensors) in logger.enum("Train", self.train_loader):
            encoder_hidden = self.encoder.init_hidden(self.device).double()

            self.optimizer.zero_grad()

            train_loss: torch.Tensor = 0
            for input_tensor, target_tensor in zip(input_tensors, target_tensors):
                encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

                train_loss += self.loss(encoder_output, target_tensor)

            train_loss.backward()
            self.optimizer.step()

            logger.store(loss=train_loss.item())
            logger.add_global_step()

    def _test(self):
        self.encoder.eval()

        with torch.no_grad():
            preds = []
            targets = []
            for input_tensors, target_tensors in logger.iterate("Test", self.test_loader):
                encoder_hidden = self.encoder.init_hidden(self.device).double()

                test_loss = 0
                for (input_tensor, target_tensor) in zip(input_tensors, target_tensors):
                    encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

                    test_loss += self.loss(encoder_output, target_tensor)
                    pred = int(encoder_output.item())

                    preds.append(pred)
                    targets.append(target_tensor.item())

            macro_f1 = f1_score(y_true=targets, y_pred=preds, average='macro')

            logger.store(test_loss=test_loss / len(self.test_loader.dataset))
            logger.store(accuracy=macro_f1)

    def _create_submission(self):
        df = pd.DataFrame(columns=['time', 'open_channels'])
        data = pd.read_csv('data/liverpool-ion-switching/test.csv')

        with torch.no_grad():
            encoder_hidden = self.encoder.init_hidden(self.device).double()
            for idx, row in data.iterrows():
                input_tensor = torch.tensor([row['signal']], dtype=torch.float64)
                encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

                pred = int(encoder_output.item())
                data = pd.DataFrame({"time": row['time'],
                                     "open_channels": pred},
                                    index=[idx])
                df = df.append(data)

        df.to_csv('data/liverpool-ion-switching/submission.csv', sep=',', index=False)

    def __call__(self):
        for _ in self.loop:
            self._train()
            self._test()
        self._create_submission()
        print('done')


class LoaderConfigs(configs.Configs):
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader


class Configs(training_loop.TrainingLoopConfigs, LoaderConfigs):
    epochs: int = 100

    loop_step = 'loop_step'
    loop_count = 'loop_count'

    is_save_models = True
    batch_size: int = 50000
    test_batch_size: int = 50000

    use_cuda: float = True
    cuda_device: int = 0
    seed: int = 5
    train_log_interval: int = 10

    is_log_parameters: bool = True

    device: any

    encoder: nn.Module

    optimizer: optim
    lr: float = 0.0002
    beta: tuple = (0.5, 0.999)
    loss = nn.MSELoss()

    set_seed = 'set_seed'

    main: RNN


@Configs.calc(Configs.encoder)
def set_encoder(c: Configs):
    encoder = EncoderRNN().to(c.device)

    return encoder.double()


@Configs.calc(Configs.optimizer)
def optimizer(c: Configs):
    return optim.Adam(c.encoder.parameters(), lr=c.lr, betas=c.beta)


@Configs.calc(Configs.set_seed)
def set_seed(c: Configs):
    torch.manual_seed(c.seed)


@Configs.calc(Configs.loop_count)
def loop_count(c: Configs):
    return c.epochs * len(c.train_loader)


@Configs.calc(Configs.loop_step)
def loop_step(c: Configs):
    return len(c.train_loader)


@Configs.calc(Configs.device)
def device(*, use_cuda, cuda_device):
    from lab.util.pytorch import get_device

    return get_device(use_cuda, cuda_device)


def _custom_dataset(is_train):
    return CsvDataset(file_path='data/liverpool-ion-switching/train.csv',
                      train=is_train,
                      test_fraction=0.1,
                      x_cols=['signal'],
                      y_cols=['open_channels'])


def _data_loader(is_train, batch_size):
    return torch.utils.data.DataLoader(_custom_dataset(is_train), batch_size=batch_size, drop_last=True)


@Configs.calc([Configs.train_loader, Configs.test_loader])
def data_loaders(c: Configs):
    train = _data_loader(True, c.batch_size)
    test = _data_loader(False, c.test_batch_size)

    return train, test


def main():
    conf = Configs()
    experiment = Experiment(writers={'sqlite', 'tensorboard'})
    experiment.calc_configs(conf,
                            {},
                            run_order=['set_seed', 'main'])
    experiment.start()
    conf.main()


if __name__ == '__main__':
    main()
