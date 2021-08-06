from typing import Optional, Dict, List, Callable, Any

import torch.optim
import torch.optim
import torch.utils.data
import torch.utils.data
from torch import nn

import labml.utils.pytorch as pytorch_utils
from labml import tracker, monit
from labml.configs import option, meta_config
from .device import DeviceConfigs
from .metrics import StateModule
from .training_loop import TrainingLoopConfigs


class ModeState:
    def __init__(self):
        self._rollback_stack = []

        self.is_train = False
        self.is_log_activations = False
        self.is_log_parameters = False
        self.is_optimize = False

    def _enter(self, mode: Dict[str, any]):
        rollback = {}
        for k, v in mode.items():
            if v is None:
                continue
            rollback[k] = getattr(self, k)
            setattr(self, k, v)

        self._rollback_stack.append(rollback)

        return len(self._rollback_stack)

    def _exit(self, n: int):
        assert n == len(self._rollback_stack)

        rollback = self._rollback_stack[-1]
        self._rollback_stack.pop(-1)

        for k, v in rollback.items():
            setattr(self, k, v)

    def update(self, *,
               is_train: Optional[bool] = None,
               is_log_parameters: Optional[bool] = None,
               is_log_activations: Optional[bool] = None,
               is_optimize: Optional[bool] = None):
        return Mode(self,
                    is_train=is_train,
                    is_log_parameters=is_log_parameters,
                    is_log_activations=is_log_activations,
                    is_optimize=is_optimize)


class Mode:
    def __init__(self, mode: ModeState, **kwargs: any):
        self.mode = mode
        self.update = {}
        for k, v in kwargs.items():
            if v is not None:
                self.update[k] = v

        self.idx = -1

    def __enter__(self):
        self.idx = self.mode._enter(self.update)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mode._exit(self.idx)


class ForwardHook:
    def __init__(self, mode: ModeState, model_name, name: str, module: torch.nn.Module):
        self.mode = mode
        self.model_name = model_name
        self.name = name
        self.module = module
        module.register_forward_hook(self)

    def save(self, name: str, output):
        if isinstance(output, torch.Tensor):
            pytorch_utils.store_var(name, output)
        elif isinstance(output, tuple):
            for i, o in enumerate(output):
                self.save(f"{name}.{i}", o)

    def __call__(self, module, i, o):
        if not self.mode.is_log_activations:
            return

        self.save(f"module.{self.model_name}.{self.name}", o)


def hook_model_outputs(mode: ModeState, model: torch.nn.Module, model_name: str = "model"):
    for name, module in model.named_modules():
        if name == '':
            name = 'full'
        ForwardHook(mode, model_name, name, module)


class Trainer:
    def __init__(self, *,
                 name: str,
                 mode: ModeState,
                 data_loader: torch.utils.data.DataLoader,
                 inner_iterations: int,
                 state_modules: List[StateModule],
                 step: Callable[[any, 'BatchIndex'], None]):
        self.mode = mode
        self.name = name
        self.step = step
        self.state_modules = state_modules
        self.__iterable = None
        self.__states = [sm.create_state() for sm in self.state_modules]
        self.inner_iterations = inner_iterations
        self.data_loader = data_loader
        self._batch_index = BatchIndex(len(self.data_loader), self.inner_iterations)

    def set_data_loader(self, data_loader: torch.utils.data.DataLoader):
        self.data_loader = data_loader
        self._batch_index = BatchIndex(len(data_loader), self.inner_iterations)
        self.__iterable = None

    def __call__(self):
        for sm, s in zip(self.state_modules, self.__states):
            sm.set_state(s)

        if self.__iterable is None or self._batch_index.completed:
            self.__iterable = iter(self.data_loader)
            self._batch_index.reset(len(self.data_loader), self.inner_iterations)
            for sm in self.state_modules:
                sm.on_epoch_start()
        with torch.set_grad_enabled(self.mode.is_train):
            self.__iterate()

        if self._batch_index.completed:
            for sm in self.state_modules:
                sm.on_epoch_end()

    def __iterate(self):
        with monit.section(self.name, is_partial=True):
            if self._batch_index.idx == 0:
                monit.progress(0)
            while not self._batch_index.iteration_completed:
                batch = next(self.__iterable)

                self.step(batch, self._batch_index)

                self._batch_index.step()
                monit.progress(self._batch_index.epoch_progress)

        self._batch_index.step_inner()


class BatchIndex:
    idx: int
    total: int
    iteration: int
    total_iterations: int

    def __init__(self, total: int, total_iterations: int):
        self.total_iterations = total_iterations
        self.total = total

    def is_interval(self, interval: int):
        if interval <= 0:
            return False
        if self.idx + 1 == self.total:
            return True
        else:
            return (self.idx + 1) % interval == 0

    @property
    def is_last(self):
        return self.idx + 1 == self.total

    @property
    def completed(self):
        return self.iteration >= self.total_iterations

    @property
    def iteration_completed(self):
        # // is important so that the last step happens on the last iteration
        return self.idx >= (self.iteration + 1) * self.total // self.total_iterations

    @property
    def epoch_progress(self):
        return self.idx / self.total

    def step(self):
        self.idx += 1

    def step_inner(self):
        self.iteration += 1

    def reset(self, total: int, total_iterations: int):
        self.total = total
        self.total_iterations = total_iterations
        self.idx = 0
        self.iteration = 0


class TrainValidConfigs(TrainingLoopConfigs):
    r"""
    This is a configurable module that you can extend for experiments that involve a 
    training and validation datasets (i.e. most DL experiments).
    This is based on :class:`labml_helpers.training_loop.TrainingLoopConfigs`.

    Arguments:
        epochs (int): Number of epochs to train on. Defaults to ``10``.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        valid_loader (torch.utils.data.DataLoader): Training data loader.
        inner_iterations (int): Number of times to switch between training and validation
         within an epoch. Defaults to ``1``.

    You can override ``init``, ``step`` functions. There is also a ``sample`` function
    that you can override to generate samples ever time it switches between training and validation.

    `Here's an example usage <https://github.com/labmlai/labml/blob/master/samples/pytorch/mnist/e_labml_helpers.py>`_.    
    """
    state_modules: List[StateModule]

    mode: ModeState

    epochs: int = 10

    trainer: Trainer
    validator: Trainer
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader

    loop_count = '_data_loop_count'
    loop_step = None

    inner_iterations: int = 1

    def init(self):
        pass

    def step(self, batch: Any, batch_idx: BatchIndex):
        raise NotImplementedError

    def run_step(self):
        for i in range(self.inner_iterations):
            with tracker.namespace('sample'):
                self.sample()
            with self.mode.update(is_train=True):
                with tracker.namespace('train'):
                    self.trainer()
            if self.validator:
                with tracker.namespace('valid'):
                    self.validator()
            tracker.save()

    def run(self):
        with monit.section("Initialize"):
            self.init()
        _ = self.validator
        _ = self.trainer
        for _ in self.training_loop:
            self.run_step()

    def sample(self):
        pass


@option(TrainValidConfigs.trainer)
def _default_trainer(c: TrainValidConfigs):
    return Trainer(name='Train',
                   mode=c.mode,
                   data_loader=c.train_loader,
                   inner_iterations=c.inner_iterations,
                   state_modules=c.state_modules,
                   step=c.step)


@option(TrainValidConfigs.validator)
def _default_validator(c: TrainValidConfigs):
    return Trainer(name='Valid',
                   mode=c.mode,
                   data_loader=c.valid_loader,
                   inner_iterations=c.inner_iterations,
                   state_modules=c.state_modules,
                   step=c.step)


@option(TrainValidConfigs.loop_count)
def _data_loop_count(c: TrainValidConfigs):
    return c.epochs


class SimpleTrainValidConfigs(TrainValidConfigs):
    r"""
    This is a configurable module that works for many standard DL experiments.
    This is based on :class:`labml_helpers.training_loop.TrainValidConfigs`.

    Arguments:
        model: A PyTorch model.
        optimizer: A PyTorch optimizer to update model.
        device: The device to train the model on. This defaults to a configurable device -
         :class:`labml_helpers.device.DeviceConfigs`.
        loss_function: A function to calculate the loss. This should accept ``model_output, target`` as
         arguments.
        update_batches (int): Number of batches to accumulate before taking an optimizer step.
         Defaults to ``1``.
        log_params_updates (int): How often (number of batches) to track model parameters and gradients.
         Defaults to a large number; i.e. logs every epoch.
        log_activations_batches (int): How often to log model activations. 
         Defaults to a large number; i.e. logs every epoch.
        log_save_batches (int): How often to call :func:`labml.tracker.save`.
    """
    optimizer: torch.optim.Adam
    model: nn.Module
    device: torch.device = DeviceConfigs()

    loss_func: nn.Module

    update_batches: int = 1
    log_params_updates: int = 2 ** 32  # 0 if not
    log_activations_batches: int = 2 ** 32  # 0 if not
    log_save_batches: int = 1

    state_modules: List[StateModule] = []

    def init(self):
        pass

    def step(self, batch: Any, batch_idx: BatchIndex):
        self.model.train(self.mode.is_train)
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(len(data))

        is_log_activations = batch_idx.is_interval(self.log_activations_batches)
        with monit.section("model"):
            with self.mode.update(is_log_activations=is_log_activations):
                output = self.model(data)

        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        if self.mode.is_train:
            with monit.section('backward'):
                loss.backward()

            if batch_idx.is_interval(self.update_batches):
                with monit.section('optimize'):
                    self.optimizer.step()
                if batch_idx.is_interval(self.log_params_updates):
                    tracker.add('model', self.model)
                self.optimizer.zero_grad()

            if batch_idx.is_interval(self.log_save_batches):
                tracker.save()


meta_config(SimpleTrainValidConfigs.update_batches,
            SimpleTrainValidConfigs.log_params_updates,
            SimpleTrainValidConfigs.log_activations_batches)


@option(SimpleTrainValidConfigs.optimizer)
def _default_optimizer(c: SimpleTrainValidConfigs):
    from labml_helpers.optimizer import OptimizerConfigs
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.model.parameters()
    return opt_conf
