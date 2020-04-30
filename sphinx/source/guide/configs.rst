Configs
=======

.. currentmodule:: lab.configs

.. code-block:: python

    import torch
    
    from lab import tracker, monit, loop, experiment, logger
    from lab.configs import BaseConfigs

The configs will be stored and in future be adjusted from
`Dashboard <https://github.com/vpj/lab_dashboard>`__

.. code-block:: python

    class DeviceConfigs(BaseConfigs):
        use_cuda: bool = True
        cuda_device: int = 0
    
        device: any

Some configs can be calculated

.. code-block:: python

    @DeviceConfigs.calc(DeviceConfigs.device)
    def cuda(c: DeviceConfigs):
        is_cuda = c.use_cuda and torch.cuda.is_available()
        if not is_cuda:
            return torch.device("cpu")
        else:
            if c.cuda_device < torch.cuda.device_count():
                return torch.device(f"cuda:{c.cuda_device}")
            else:
                logger.log(f"Cuda device index {c.cuda_device} higher than "
                           f"device count {torch.cuda.device_count()}", Text.warning)
                return torch.device(f"cuda:{torch.cuda.device_count() - 1}")

Configs classes can be inherited. This can be used to separate configs
into module and it is quite neat when you want to inherit entire
experiment setups and make a few modifications.

.. code-block:: python

    class Configs(DeviceConfigs):
        model_size: int = 10
            
        model: any = 'cnn_model'

You can specify multiple config calculator functions. The function given
by the string for respective attribute will be picked.

.. code-block:: python

    @Configs.calc(Configs.model)
    def cnn_model(c: Configs):
        return c.model_size * 10
    
    @Configs.calc(Configs.model)
    def lstm_model(c: Configs):
        return c.model_size * 2

The experiment will calculate the configs.

.. code-block:: python

    conf = Configs()
    conf.model = 'lstm_model'
    experiment.create(name='test_configs')
    experiment.calculate_configs(conf)
    logger.inspect(model=conf.model)



.. raw:: html

    <pre>                                                                                                    
    <span style="color: #60C6C8">model: </span><strong>20</strong></pre>


.. code-block:: python

    experiment.start()



.. raw:: html

    <pre>
    <strong><span style="text-decoration: underline">test_configs</span></strong>: <span style="color: #208FFB">f7dce148895e11ea944cacde48001122</span>
    	[dirty]: <strong><span style="color: #DDB62B">"lint fixed"</span></strong>
    <span style="text-decoration: underline">Configs:</span>
    	<span style="color: #60C6C8">cuda_device</span><span style="color: #C5C1B4"> = </span><strong>0</strong>	
    	<span style="color: #60C6C8">device</span><span style="color: #C5C1B4"> = </span><strong>cpu</strong>	<span style="color: #C5C1B4">cuda</span>
    	<span style="color: #60C6C8"><strong><span style="color: #DDB62B">model</span></strong></span><span style="color: #C5C1B4"> = </span><strong>20</strong>	lstm_model<span style="color: #C5C1B4">	[</span>cnn_model<span style="color: #C5C1B4">]</span>
    	<span style="color: #60C6C8">model_size</span><span style="color: #C5C1B4"> = </span><strong>10</strong>	
    	<span style="color: #60C6C8">use_cuda</span><span style="color: #C5C1B4"> = </span><strong>True</strong>	
    </pre>


