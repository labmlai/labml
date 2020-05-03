from typing import Optional, Callable

import torch.optim
import torch.utils.data
from torch import nn

import lab.utils.pytorch as pytorch_utils
from lab import tracker, monit, loop
from lab.helpers.training_loop import TrainingLoopConfigs
from lab.utils.pytorch import get_device


class Trainer:
    def __init__(self, *,
                 name: str,
                 model: nn.Module,
                 optimizer: Optional[torch.optim.Adam],
                 loss_func: Callable,
                 accuracy_func: Callable,
                 data_loader: torch.utils.data.DataLoader,
                 is_increment_global_step: bool,
                 log_interval: Optional[int]):
        r"""
        Arguments:
            loss_func(Callable): A module with a call signature
                ``(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor``
            accuracy_func(Callable): A module with a call signature
                ``(output: torch.Tensor, target: torch.Tensor) -> int``
        """
        self.accuracy_func = accuracy_func
        self.loss_func = loss_func
        self.log_interval = log_interval
        self.is_increment_global_step = is_increment_global_step
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.name = name
        self.model = model

        tracker.set_queue(".loss", 20, True)
        tracker.set_scalar(".accuracy", True)

    def __call__(self):
        if self.optimizer is not None:
            self.model.train()
            self.iterate()
        else:
            self.model.eval()
            with torch.no_grad():
                self.iterate()

    def iterate(self):
        device = get_device(self.model)
        correct_sum = 0
        total_samples = 0

        for i, (data, target) in monit.enum(self.name, self.data_loader):
            data, target = data.to(device), target.to(device)

            if self.optimizer is not None:
                self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss_func(output, target)
            correct_sum += self.accuracy_func(output, target)
            total_samples += len(target)

            tracker.add(".loss", loss)

            if self.optimizer is not None:
                loss.backward()
                self.optimizer.step()

            if self.is_increment_global_step:
                loop.add_global_step(len(target))

            if self.log_interval is not None and (i + 1) % self.log_interval == 0:
                tracker.save()

        tracker.add(".accuracy", correct_sum / total_samples)


class TrainValidConfigs(TrainingLoopConfigs):
    epochs: int = 10

    loss_func: Callable
    accuracy_func: Callable
    optimizer: torch.optim.Adam
    model: nn.Module
    trainer: Trainer
    validator: Trainer

    train_log_interval: int = 10

    loop_count = 'data_loop_count'
    loop_step = 'data_loop_step'

    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader

    is_log_parameters: bool = True

    def run(self):
        if self.is_log_parameters:
            pytorch_utils.add_model_indicators(self.model)

        for _ in self.training_loop:
            with tracker.namespace('train'):
                self.trainer()
            with tracker.namespace('valid'):
                self.validator()
            if self.is_log_parameters:
                pytorch_utils.store_model_indicators(self.model)


@TrainValidConfigs.calc(TrainValidConfigs.trainer)
def trainer(c: TrainValidConfigs):
    return Trainer(name='Train',
                   model=c.model,
                   optimizer=c.optimizer,
                   loss_func=c.loss_func,
                   accuracy_func=c.accuracy_func,
                   data_loader=c.train_loader,
                   is_increment_global_step=True,
                   log_interval=c.train_log_interval)


@TrainValidConfigs.calc(TrainValidConfigs.validator)
def validator(c: TrainValidConfigs):
    return Trainer(name='Valid',
                   model=c.model,
                   optimizer=None,
                   loss_func=c.loss_func,
                   accuracy_func=c.accuracy_func,
                   data_loader=c.valid_loader,
                   is_increment_global_step=False,
                   log_interval=None)


@TrainValidConfigs.calc(TrainValidConfigs.loop_count)
def data_loop_count(c: TrainValidConfigs):
    return c.epochs * len(c.train_loader.dataset)


@TrainValidConfigs.calc(TrainValidConfigs.loop_step)
def data_loop_step(c: TrainValidConfigs):
    return len(c.train_loader.dataset)
