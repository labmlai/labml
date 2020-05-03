
MNIST Classification
====================

This tutorial’ll help and guide you to add lab features to your machine
learning project. We’ll be using
`MNSIT <http://yann.lecun.com/exdb/mnist/>`__ dataset and simple a
`convolutional neural network
(CNN) <https://en.wikipedia.org/wiki/Convolutional_neural_network/>`__
to build our model.

You can find the complete code by following this
`link <https://github.com/vpj/lab/blob/master/samples/mnist_loop.py/>`__.

Model
-----

In this section, we’ll build a simple CNN model using PyTorch to
classify MNIST images into 10 classes.

Imports
~~~~~~~

We’ll want to start with importing the PyTorch libraries as well as the
Lab. Note that we’ll be using the following imports not only for this
tutorial but also throughout this tutorial series.

.. code-block:: python

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
    from lab.configs import BaseConfigs

Model Architecture
~~~~~~~~~~~~~~~~~~

We’ll build the following CNN model in this tutorial.

.. raw:: html

  <p align="center">
    <img style="max-width:100%;"
    src="../_static/img/cnn.png"
    width="1024" title="CNN">
  </p>

Model Implementation
~~~~~~~~~~~~~~~~~~~~

PyTorch makes it pretty easy to implement a simple CNN.

.. code-block:: python

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

Model Configs
-------------

Lab makes it easier to separate configs from the model implementation
and allows you to maintain a clean and reusable code. We’ll first define
the ``Configs Class`` with a few config parameters. This class should be
inherited from :class:`lab.configs.BaseConfigs` class.

Configs Class
~~~~~~~~~~~~~

.. code-block:: python

    class Configs(BaseConfigs):
        epochs: int = 10
    
        batch_size: int = 64
        test_batch_size: int = 1000
    
        model: nn.Module
    
        learning_rate: float = 0.01
        optimizer: optim.SGD
    
        device: any
        use_cuda: bool = True
        cuda_device: int = 0

Here, we have defined our training and test ``batch_sizes``, the number
of ``epochs`` and the ``learning_rate``. Note that we have only defined
the type of ``optimizer``, ``model`` and ``device``.

Adding Configs
~~~~~~~~~~~~~~

We’ll define our ``model function`` as below and use
:func:`lab.configs.BaseConfigs.calc` to modify it. We’ll be using the
model that is implemented in the previous section. With the
:func:`lab.configs.BaseConfigs.calc` decorator, Lab identifies and add
to the ``Configs`` in run time.

.. code-block:: python

    @Configs.calc(Configs.model)
    def model(c: Configs):
        m: Net = Net()
        m.to(c.device)
        return m

Next, we’ll define our optimization algorithm. In this case, we’ll be
using `Adam <https://arxiv.org/pdf/1412.6980.pdf>`__, which is an
extension to stochastic gradient descent.

.. code-block:: python

    @Configs.calc(Configs.optimizer)
    def sgd_optimizer(c: Configs):
        return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)

We can specify the ``device`` using
:func:`lab.util.pytorch.get_device`.

.. code-block:: python

    @Configs.calc(Configs.device)
    def device(c: Configs):
        from lab.util.pytorch import get_device
    
        return get_device(c.use_cuda, c.cuda_device)

Data Loaders
~~~~~~~~~~~~

Define the ``data_loader`` method as follows. Here, we utilise the
`torch
DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__,
and
`MNIST <https://pytorch.org/docs/stable/torchvision/datasets.html#mnist>`__
dataset from PyTorch.

.. code-block:: python

    def _data_loader(is_train, batch_size):
        return torch.utils.data.DataLoader(
            datasets.MNIST(str(logger.get_data_path()),
                            train=is_train,
                            download=True,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            batch_size=batch_size, shuffle=True)

.. code-block:: python

    class LoaderConfigs(BaseConfigs):
        train_loader: torch.utils.data.DataLoader
        test_loader: torch.utils.data.DataLoader

We have created the ``LoaderConfigs`` class by inheriting
:class:`lab.configs.BaseConfigs`. Therefore, your main ``Configs``
class now can be inherited from ``LoaderConfigs``.

.. code-block:: python

    class Configs(LoaderConfigs):
        epochs: int = 10

This can be used to separate ``configs`` into modules and it is quite
neat when you want to inherit entire experiment setups and make a few
modifications.

Training Loop Configs
~~~~~~~~~~~~~~~~~~~~~

You can inherit your ``Configs`` class from
:class:`lab.helpers.training_loop.TrainingLoopConfigs` and change few
related configs accordingly.

.. code-block:: python

    class Configs(TrainingLoopConfigs):
        loop_step = 'loop_step'
        loop_count = 'loop_count'
        is_save_models: bool = False
    
    
    @Configs.calc(Configs.loop_count)
    def loop_count(c: Configs):
        return c.epochs * len(c.train_loader)
    
    
    @Configs.calc(Configs.loop_step)
    def loop_step(c: Configs):
        return len(c.train_loader)


Model Training
--------------

In this section, We’ll describe about model training.

Passing Configs
~~~~~~~~~~~~~~~

First, we define a separate class named ``MNIST`` for model training,
and then pass the ``configs`` that we defined in the previous section.

.. code-block:: python

    class MNIST:
        def __init__(self, c: 'Configs'):
            self.model = c.model
            self.device = c.device
            self.train_loader = c.train_loader
            self.test_loader = c.test_loader
            self.optimizer = c.optimizer
            self.train_log_interval = c.train_log_interval
            self.loop = c.training_loop
            self.__is_log_parameters = c.is_log_parameters

Training Iterations
~~~~~~~~~~~~~~~~~~~

Let’s add training iterations as a separate method.

.. code-block:: python

    def train(self):
        self.model.train()
        for i, (data, target) in monit.enum("Train", self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
    
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
    
            loop.add_global_step()

We have utilised the :func:`lab.monit.enum` to iterate thorough the
dataset. Moreover, we call the :func:`lab.loop.add_global_step` inside
the ``iterator`` to increment the number of ``global step by one``.
Furthermore, you may need to log metrics to track your model performance
in each iteration.

In the following code snippet, We are logging ``train_loss`` in each
iteration. :func:`lab.tracker.add` method stores values (as ``Sclars``
by default) of each metric for each iteration.
:func:`lab.tracker.save` writes each stored metric (this can be called
in a predefined log interval) and then free up the memory.

.. code-block:: python

    self.optimizer.step()
    
    loop.add_global_step()
    logger.add_global_step()
    
    if i % self.train_log_interval == 0:
            tracker.save()

Training Loop
~~~~~~~~~~~~~

Next, we need to go through a few iterations of the entire dataset (few
epochs). For this purpose, we can utilise :func:`lab.loop.loop` method
as follows. Note that configuration of the ``training_loop`` was
discussed in the previous section.

.. code-block:: python

     def __call__(self):
        for _ in self.training_loop:
            self.train()
            self.test()
            if self.is_log_parameters:
                pytorch_utils.store_model_indicators(self.model)

In the above code snippet, we make use of the python magic method
``__call__``.

Logging Model Indicators
~~~~~~~~~~~~~~~~~~~~~~~~

If you need to log model indicators such as biases, weights and gradient
values of the model in each iteration, Lab provides very continent
method via :func:`lab.utils.pytorch.add_model_indicators`.

.. code-block:: python

    def __call__(self):
        pytorch_utils.add_model_indicators(self.model)

Logging Indicators
~~~~~~~~~~~~~~~~~~

Without specifying, :func:`lab.tracker.add` store metric values as
``Scalars``. However, if you need to add a metric value as a
:class:`lab.tracker.set_histogram` or
:class:`lab.tracker.set_queue`, you need to provide the type
beforehand. Let’s define the type of our ``train_loss`` metric as a
``Histogram``.

.. code-block:: python

    tracker.set_histogram("train_loss", 20, True)
    
    for _ in self.training_loop:
         self.train()

Experiment
----------

As the final step, you need to start and run the experiment. Lab
provides a convenient way to do this.

.. code-block:: python

    def run():
        conf = Configs()
        experiment.create(writers={'sqlite', 'tensorboard'})
        experiment.calculate_configs(conf,
                                     {},
                                     ['set_seed', 'run'])
        experiment.add_pytorch_models(dict(model=conf.model))
        experiment.start()
        conf.main()
    
    def main():
        run()
    
    if __name__ == '__main__':
        main()

Note that in the above code snippet, We have declared an
:class:`lab.experiment` and passed the ``writers``, in this case,
``sqlite`` and ``tensorboard``. By default Lab’ll writes every log to
the console. Moreover, you can pass the order of calculating ``configs``
by passing a list of the order in :func:`lab.experiment.calc_configs`.

Hyper-parameter Tuning
----------------------

For any machine learning model, it’s paramount important to find out the
best set of ``hyperparameters`` that improves the model metrics.
Usually, this is done experimentally and iteratively. Lab provides a
nice way to separate your ``hyperparameters`` and browse via
lab-dashboard.

Let’s find out the best set of ``kernel_sizes`` for our model. In order
to do that, we first need to change the model implementation as below.

.. code-block:: python

    class Net(nn.Module):
        def __init__(self, conv1_kernal, conv2_kernal):
            super().__init__()
            self.size = (28 - conv1_kernal - 2 * conv2_kernal + 3) // 4
    
            self.conv1 = nn.Conv2d(1, 20, conv1_kernal, 1)
            self.conv2 = nn.Conv2d(20, 50, conv2_kernal, 1)
            self.fc1 = nn.Linear(self.size * self.size * 50, 500)
            self.fc2 = nn.Linear(500, 10)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, self.size * self.size * 50)
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    
    
    class Configs(TrainingLoopConfigs, LoaderConfigs):
        conv1_kernal: int
        conv2_kernal: int
    
    @Configs.calc(Configs.model)
    def model(c: Configs):
        m: Net = Net(c.conv1_kernal, c.conv2_kernal)
        m.to(c.device)
        return m

It’s important to note that ``input_size`` of ``fc1`` is changing based
on the ``kernel_sizes`` of two convolutions.

Moreover, you can run a simple grid search as below.

.. code-block:: python

    def run(hparams: dict):
        loop.set_global_step(0)
    
        conf = Configs()
        experiment.create(name='mnist_hyperparam_tuning', writers={'sqlite', 'tensorboard'})
        experiment.calculate_configs(conf,
                                     hparams,
                                     ['set_seed', 'main'])
        experiment.add_pytorch_models(dict(model=conf.model))
        experiment.start()
    
        conf.main()
    
    
    def main():
        for conv1_kernal in [3, 5]:
            for conv2_kernal in [3, 5]:
                hparams = {
                    'conv1_kernal': conv1_kernal,
                    'conv2_kernal': conv2_kernal,
                }
    
                run(hparams)

Lab, by default identifies the parameters that passes to
:func:`lab.experiment.calculate_configs` as ``hyperparameters`` and
treat them accordingly.

