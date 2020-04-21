Model Configs
=============

Lab makes it easier to separate configs from the model implementation and allows you to maintain a clean and reusable code.
We'll first define the `Configs` class with a few config parameters. `Configs Class` should be inherited from Lab `configs.Configs` class.

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

Here, we have defined our training and test `batch_sizes`, the number of `epochs` and the `learning_rate`. Note that we have only defined the type of `optimizer`, `model` and the `device`.

Adding Configs
--------------

We'll define our `model` as a lab `config.calc` function. We will be using the model implemented in the previous section. With the `@Configs.calc` decorator, Lab will identify and add to the Configs in run time.

.. code-block:: python

   @Configs.calc(Configs.model)
        def model(c: Configs):
            m: Net = Net()
            m.to(c.device)
            return m

Next, we'll define our optimization algorithm. In this case, we will be using Adam, which is an extension to stochastic gradient descent.

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

Define the `data_loader` method as follows. Here, we utilise the `torch DataLoader` and the `MNIST` dataset from  `torchvision datasets`.

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

We have created the `LoaderConfigs` class by inheriting from `configs.Configs` class. Therefore, your main   `Configs` class now can be inherited from `LoaderConfigs`.

.. code-block:: python

   class Configs(LoaderConfigs):
        epochs: int = 10


This can be used to separate configs into modules and it is quite neat when you want to inherit entire experiment setups and make a few modifications.

Training Loop Configs
---------------------

You can inherit your `Configs` class from Lab `TrainingLoopConfigs` and change few related configs accordingly.

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