### Configs


```python
from lab import configs
```

The configs will be stored and in future be adjusted from  [Dashboard](https://github.com/vpj/lab_dashboard)


```python
class DeviceConfigs(configs.Configs):
    use_cuda: bool = True
    cuda_device: int = 0

    device: any
```

Some configs can be calculated


```python
import torch
```


```python
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
```

Configs classes can be inherited. This can be used to separate configs into module and it is quite neat when you want to inherit entire experiment setups and make a few modifications. 


```python
class Configs(DeviceConfigs):
    model_size: int = 10
        
    model: any = 'cnn_model'
```

You can specify multiple config calculator functions. The function given by the string for respective attribute will be picked.


```python
@Configs.calc(Configs.model)
def cnn_model(c: Configs):
    return c.model_size * 10

@Configs.calc(Configs.model)
def lstm_model(c: Configs):
    return c.model_size * 2
```

The experiment will calculate the configs.


```python
conf = Configs()
conf.model = 'lstm_model'
experiment = Experiment(name='test_configs')
experiment.calc_configs(conf)
logger.info(model=conf.model)
```


<pre>                                                                                                    
<span style="color: #60C6C8">model: </span><strong>20</strong></pre>