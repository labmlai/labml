Model Training
==============

In this section, We'll describe about model training.

Passing Configs
---------------

First, we define a separate class named `MNIST` for model training, and then pass the configs that we defined in the previous section.

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


We have utilised the Lab `logger.enum` to iterate thorough the dataset. Moreover, we call the `logger.add_global_step()` method inside the iterator to increase the number of global step by one. Furthermore, you may need to log matrices to track your model performance in each iteration.

In the following code snippet, We are logging `train_loss` in each iteration. Lab `logger.store` method stores values (as Sclars by default) of each metric for each iteration. `logger.write()` writes each stored metric (this can be called in a predefined log interval) and then free up the memory.

.. code-block:: python
    self.optimizer.step()

    logger.store(train_loss=loss)
    logger.add_global_step()

    if i % self.train_log_interval == 0:
        logger.write()


Training Loop
-------------

Next, we need to go through a few iterations of the entire dataset (few epochs). For this purpose, we can utilise `logger.loop` method as follows. Note that configuration of the `training_loop` was discussed in the previous section.

.. code-block:: python

    def __call__(self):
        logger_util.add_model_indicators(self.model)

        for _ in self.loop:
            self._train()
            self._test()
            self.__log_model_params()

In the above code snippet, we make use of the python magic method ` __call__`.

Logging Model Indicators
------------------------

If you need to log model indicators such as biases, weights and gradient values of the model in each iteration, Lab provides  very continent method via `logger_util.add_model_indicators`.

.. code-block:: python

   def run(self):
       logger_util.add_model_indicators(self.model)


Logging Indicators
------------------

Without specifying, `logger.store` method store metric values as Scalars. However, if you need to store a metric value as a Histograms or Queues, you need to provide the type beforehand. Let's define the type of our `train_loss` metric as a Histogram.


.. code-block:: python

   logger.add_indicator(Histogram("train_loss"True))

   for _ in self.loop:
        self._train()