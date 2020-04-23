MNIST Classification
********************

Model
=====

This tutorial will describe and guide you to add lab features to your machine learning project. We will be using `MNSIT <http://yann.lecun.com/exdb/mnist/>`_  dataset and simple
`convolutional neural network (CNN) <https://en.wikipedia.org/wiki/Convolutional_neural_network/>`_ to build our model.

You can find the complete code by following this `link <https://github.com/vpj/lab/blob/master/samples/mnist_loop.py/>`_.

In this section, we'll build a simple CNN model using PyTorch to classify MNIST images into 10 classes.

Imports
-------

We’ll want to start with importing the PyTorch libraries as well as the Lab. Note that we will be using the following imports not only for this tutorial but also throughout this tutorial series.

.. code-block:: python

    import torch.optim as optim
    import torch.utils.data
    from torchvision import datasets, transforms

    from lab import logger, configs
    from lab import training_loop
    from lab.experiment.pytorch import Experiment
    from lab.logger.indicators import Queue, Histogram
    from lab.logger.util import pytorch as logger_util

Model Architecture
------------------

We'll build the following CNN model in this tutorial.

.. raw:: html

    <p align="center">
      <img style="max-width:100%;"
       src="../_static/img/cnn.png"
       width="1024" title="CNN">
    </p>

Model Implementation
--------------------

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
=============

Lab makes it easier to separate configs from the model implementation and allows you to maintain a clean and reusable code.
We'll first define the ``Configs Class`` with a few config parameters. This class should be inherited from :py:class:`lab.configs.Configs` class.

Configs Class
-------------

.. code-block:: python

  class Configs(configs.Configs):
        epochs: int = 10

        batch_size: int = 64
        test_batch_size: int = 1000

        model: nn.Module

        learning_rate: float = 0.01
        optimizer: optim.SGD

        device: any
        use_cuda: bool = True
        cuda_device: int = 0

Here, we have defined our training and test ``batch_sizes``, the number of ``epochs`` and the ``learning_rate``. Note that we have only defined the type of ``optimizer``, ``model`` and the ``device``.

Adding Configs
--------------

We'll define our ``model function`` as below and use :py:func:`lab.configs.Configs.calc` to modify it. We'll be using the model that is implemented in the previous section. With the :py:func:`lab.configs.Configs.calc` decorator, Lab identifies and add to the ``Configs`` in run time.

.. code-block:: python

   @Configs.calc(Configs.model)
        def model(c: Configs):
            m: Net = Net()
            m.to(c.device)
            return m

Next, we'll define our optimization algorithm. In this case, we'll be using `Adam <https://arxiv.org/pdf/1412.6980.pdf>`_, which is an extension to stochastic gradient descent.

.. code-block:: python

   @Configs.calc(Configs.optimizer)
        def sgd_optimizer(c: Configs):
            return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)

We can specify the device as follows.

.. code-block:: python

    @Configs.calc(Configs.optimizer)
    def sgd_optimizer(c: Configs):
        return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)


Data Loaders
------------

Define the ``data_loader`` method as follows. Here, we utilise the `torch DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_, and `MNIST <https://pytorch.org/docs/stable/torchvision/datasets.html#mnist>`_ dataset from PyTorch.

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

   class LoaderConfigs(configs.Configs):
        train_loader: torch.utils.data.DataLoader
        test_loader: torch.utils.data.DataLoader

We have created the ``LoaderConfigs`` class by inheriting :py:class:`lab.configs.Configs`. Therefore, your main ``Configs`` class now can be inherited from ``LoaderConfigs``.

.. code-block:: python

   class Configs(LoaderConfigs):
        epochs: int = 10


This can be used to separate configs into modules and it is quite neat when you want to inherit entire experiment setups and make a few modifications.

Training Loop Configs
---------------------

You can inherit your ``Configs`` class from :py:class:`lab.training_loop.TrainingLoopConfigs` and change few related configs accordingly.

.. code-block:: python

  class Configs(configs.Configs, training_loop.TrainingLoopConfigs):

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
==============

In this section, We'll describe about model training.

Passing Configs
---------------

First, we define a separate class named ``MNIST`` for model training, and then pass the configs that we defined in the previous section.

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
-------------------

Let's add training iterations as a separate method.

.. code-block:: python

    self.model.train()
    for i, (data, target) in logger.enum("Train", self.train_loader):
        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()

        logger.add_global_step()


We have utilised the :py:func:`lab.logger.enum` to iterate thorough the dataset. Moreover, we call the :py:func:`lab.logger.add_global_step` inside the ``iterator`` to increase the number of ``global step by one``. Furthermore, you may need to log metrics to track your model performance in each iteration.

In the following code snippet, We are logging ``train_loss`` in each iteration. :py:func:`lab.logger.store` method stores values (as ``Sclars`` by default) of each metric for each iteration. :py:func:`lab.logger.write` writes each stored metric (this can be called in a predefined log interval) and then free up the memory.

.. code-block:: python

    self.optimizer.step()

    logger.store(train_loss=loss)
    logger.add_global_step()

    if i % self.train_log_interval == 0:
        logger.write()


Training Loop
-------------

Next, we need to go through a few iterations of the entire dataset (few epochs). For this purpose, we can utilise :py:func:`lab.logger.loop` method as follows. Note that configuration of the ``training_loop`` was discussed in the previous section.

.. code-block:: python

    def __call__(self):
        logger_util.add_model_indicators(self.model)

        for _ in self.loop:
            self._train()
            self._test()
            self.__log_model_params()

In the above code snippet, we make use of the python magic method ``__call__``.

Logging Model Indicators
------------------------

If you need to log model indicators such as biases, weights and gradient values of the model in each iteration, Lab provides very continent method via :py:func:`logger_util.add_model_indicators`.

.. code-block:: python

   def run(self):
       logger_util.add_model_indicators(self.model)


Logging Indicators
------------------

Without specifying, :py:func:`lab.logger.store` store metric values as Scalars. However, if you need to store a metric value as a  :py:class:`lab.logger.indicators.Histogram` or :py:class:`lab.logger.indicators.Queue`, you need to provide the type beforehand. Let's define the type of our ``train_loss`` metric as a :py:class:`lab.logger.indicators.Histogram`.


.. code-block:: python

   logger.add_indicator(Histogram("train_loss"True))

   for _ in self.loop:
        self._train()

Experiment
==========

As the final step, you need to start and run the experiment. Lab provides a convenient way to do this.


.. code-block:: python

    def main():
        conf = Configs()
        experiment = Experiment(writers={'sqlite', 'tensorboard'})
        experiment.calc_configs(conf,
                                {'optimizer': 'adam_optimizer'},
                                ['set_seed', 'run'])
        experiment.add_models(dict(model=conf.model))
        experiment.start()
        conf.main()


    if __name__ == '__main__':
        main()

Note that in the above code snippet, We have declared an :py:class:`lab.experiment.pytorch.Experiment` and passed the writers, in this case, ``sqlite`` and ``tensorboard``. By default Lab'll writes every log to the console. Moreover, you can pass the order of calculating configs by passing a list of the order in :py:func:`lab.experiment.Experiment.calc_configs`.

Hyper-parameter Tuning
======================

Analytics
=========

