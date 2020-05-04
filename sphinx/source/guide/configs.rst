Configs
=======

.. currentmodule:: lab.configs

The configs will be stored and in future be adjusted from
`Dashboard <https://github.com/vpj/lab_dashboard>`__

.. code-block:: python

    import torch
    from torch import nn
    
    from lab import tracker, monit, loop, experiment, logger
    from lab.configs import BaseConfigs

Here is how you define a configurations class.

.. code-block:: python

    class DeviceConfigs(BaseConfigs):
        use_cuda: bool = True
        cuda_device: int = 0
    
        device: any

Some configs can be calculated.

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

Configs classes can be inherited. This lets you separate configs into
modules instead of passing `monolithic config
object <https://www.reddit.com/r/MachineLearning/comments/g1vku4/d_antipatterns_in_open_sourced_ml_research_code/>`__
and it is quite neat when you want to inherit an entire experiment
setups and make a few modifications.

.. code-block:: python

    class Configs(DeviceConfigs):
        model_size: int = 1024
        input_size: int = 10
        output_size: int = 10
            
        model: any = 'two_hidden_layer'

You can specify multiple config calculator functions. The function given
by the string for respective attribute will be picked.

.. code-block:: python

    class OneHiddenLayerModule(nn.Module):
        def __init__(self, input_size: int, model_size: int, output_size: int):
            super().__init__()
            self.input_fc = nn.Linear(input_size, model_size)
            self.output_fc = nn.Linear(model_size, output_size)
        
        def forward(x: torch.Tensor):
            x = F.relu(self.input_fc(x))
            return self.output_fc(x)
        
    # This is just for illustration purposes, ideally you should have a configeration option
    # for number of hidden layers.
    # A real world example would be different architectures, like a dense network vs a CNN
    class TwoHiddenLayerModule(nn.Module):
        def __init__(self, input_size: int, model_size: int, output_size: int):
            super().__init__()
            self.input_fc = nn.Linear(input_size, model_size)
            self.middle_fc = nn.Linear(model_size, model_size)
            self.output_fc = nn.Linear(model_size, output_size)
        
        def forward(x: torch.Tensor):
            x = F.relu(self.input_fc(x))
            x = F.relu(self.middle_fc(x))
            return self.output_fc(x)
    
    
    @Configs.calc(Configs.model)
    def one_hidden_layer(c: Configs):
        return OneHiddenLayerModule(c.input_size, c.model_size, c.output_size)
    
    @Configs.calc(Configs.model)
    def two_hidden_layer(c: Configs):
        return TwoHiddenLayerModule(c.input_size, c.model_size, c.output_size)

Note that the configurations calculators pass only the needed parameters
and not the whole config object. The library forces you to do that.

However, you can directly set the model as an option, with ``__init__``
accepting ``Configs`` as a parameter, it is not a usage pattern we
encourage.

Here’s how you run an experiment with the configurations.

.. code-block:: python

    conf = Configs()
    conf.model = 'one_hidden_layer'
    experiment.create(name='test_configs')
    experiment.calculate_configs(conf)
    logger.inspect(model=conf.model)



.. raw:: html

    <pre>
    <span style="color: #60C6C8">model: </span><strong>OneHiddenLayerModule(</strong>
    <strong>  (input_fc): Linear(in_features=10, out_features=1024, bi ...</strong></pre>


.. code-block:: python

    experiment.start()



.. raw:: html

    <pre>
    <strong><span style="text-decoration: underline">test_configs</span></strong>: <span style="color: #208FFB">6233df648dc511eaa65eacde48001122</span>
    	[dirty]: <strong><span style="color: #DDB62B">"Merge pull request #33 from lab-ml/tutorial</span></strong>
    <strong><span style="color: #DDB62B"></span></strong>
    <strong><span style="color: #DDB62B">Just changed as for the new API"</span></strong>
    <span style="text-decoration: underline">Configs:</span>
    	<span style="color: #60C6C8">cuda_device</span><span style="color: #C5C1B4"> = </span><strong>0</strong>	
    	<span style="color: #60C6C8">device</span><span style="color: #C5C1B4"> = </span><strong>cpu</strong>	<span style="color: #C5C1B4">cuda</span>
    	<span style="color: #60C6C8">input_size</span><span style="color: #C5C1B4"> = </span><strong>10</strong>	
    	<span style="color: #60C6C8"><strong><span style="color: #DDB62B">model</span></strong></span><span style="color: #C5C1B4"> = </span><strong>OneHiddenLayerModule(  (input_fc): Linea...</strong>	one_hidden_layer<span style="color: #C5C1B4">	[</span>two_hidden_layer<span style="color: #C5C1B4">]</span>
    	<span style="color: #60C6C8">model_size</span><span style="color: #C5C1B4"> = </span><strong>1024</strong>	
    	<span style="color: #60C6C8">output_size</span><span style="color: #C5C1B4"> = </span><strong>10</strong>	
    	<span style="color: #60C6C8">use_cuda</span><span style="color: #C5C1B4"> = </span><strong>True</strong>	
    </pre>

