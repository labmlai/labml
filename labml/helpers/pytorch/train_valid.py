from typing import Optional, Callable

import labml.utils.pytorch as pytorch_utils
import numpy as np
import torch.optim
import torch.utils.data
from labml import tracker, monit
from labml.configs import option
from labml.helpers.training_loop import TrainingLoopConfigs
from labml.utils.pytorch import get_device
from torch import nn


class BatchStep:
    def prepare_for_iteration(self):
        raise NotImplementedError()

    def init_stats(self):
        return {}

    def update_stats(self, stats: any, update: any):
        for k, v in update.items():
            if k not in stats:
                stats[k] = []
            stats[k].append(v)

    def log_stats(self, stats: any):
        raise NotImplementedError()

    def process(self, batch: any):
        raise NotImplementedError()


class SimpleBatchStep(BatchStep):
    def __init__(self, *,
                 model: nn.Module,
                 optimizer: Optional[torch.optim.Adam],
                 loss_func: Callable,
                 accuracy_func: Callable):
        self.accuracy_func = accuracy_func
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.model = model

        tracker.set_queue(".loss", 20, True)
        if self.accuracy_func is not None:
            tracker.set_scalar(".accuracy", True)

    def log_stats(self, stats: any):
        if self.accuracy_func is not None:
            tracker.add(".accuracy", np.sum(stats['correct']) / np.sum(stats['samples']))

    def prepare_for_iteration(self):
        if self.optimizer is None:
            self.model.eval()
            return True
        else:
            self.model.train()
            return False

    def process(self, batch: any):
        device = get_device(self.model)
        data, target = batch
        data, target = data.to(device), target.to(device)
        stats = {
            'samples': len(data)
        }

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        output = self.model(data)
        loss = self.loss_func(output, target)
        if self.accuracy_func is not None:
            stats['correct'] = self.accuracy_func(output, target)

        tracker.add(".loss", loss)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()

        return stats


class Trainer:
    def __init__(self, *,
                 name: str,
                 batch_step: BatchStep,
                 data_loader: torch.utils.data.DataLoader,
                 is_increment_global_step: bool,
                 log_interval: Optional[int]):
        self.batch_step = batch_step
        self.log_interval = log_interval
        self.is_increment_global_step = is_increment_global_step
        self.data_loader = data_loader
        self.name = name

    def __call__(self):
        if self.batch_step.prepare_for_iteration():
            with torch.no_grad():
                self.iterate()
        else:
            self.iterate()

    def iterate(self):
        stats = self.batch_step.init_stats()

        for i, batch in monit.enum(self.name, self.data_loader):
            update = self.batch_step.process(batch)
            self.batch_step.update_stats(stats, update)

            if self.is_increment_global_step:
                tracker.add_global_step(update['samples'])

            if self.log_interval is not None and (i + 1) % self.log_interval == 0:
                tracker.save()

        self.batch_step.log_stats(stats)


class TrainValidConfigs(TrainingLoopConfigs):
    epochs: int = 10

    loss_func: Callable
    accuracy_func: Callable
    optimizer: torch.optim.Adam
    model: nn.Module
    train_batch_step: BatchStep = 'simple_train_batch_step'
    valid_batch_step: BatchStep = 'simple_valid_batch_step'

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


@option(TrainValidConfigs.train_batch_step)
def simple_train_batch_step(c: TrainValidConfigs):
    return SimpleBatchStep(model=c.model,
                           optimizer=c.optimizer,
                           loss_func=c.loss_func,
                           accuracy_func=c.accuracy_func)


@option(TrainValidConfigs.valid_batch_step)
def simple_valid_batch_step(c: TrainValidConfigs):
    return SimpleBatchStep(model=c.model,
                           optimizer=None,
                           loss_func=c.loss_func,
                           accuracy_func=c.accuracy_func)


@option(TrainValidConfigs.trainer)
def trainer(c: TrainValidConfigs):
    return Trainer(name='Train',
                   batch_step=c.train_batch_step,
                   data_loader=c.train_loader,
                   is_increment_global_step=True,
                   log_interval=c.train_log_interval)


@option(TrainValidConfigs.validator)
def validator(c: TrainValidConfigs):
    return Trainer(name='Valid',
                   batch_step=c.valid_batch_step,
                   data_loader=c.valid_loader,
                   is_increment_global_step=False,
                   log_interval=None)


@option(TrainValidConfigs.loop_count)
def data_loop_count(c: TrainValidConfigs):
    return c.epochs * len(c.train_loader.dataset)


@option(TrainValidConfigs.loop_step)
def data_loop_step(c: TrainValidConfigs):
    return len(c.train_loader.dataset)
