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

