<p align="center">
  <img src="https://github.com/vpj/lab/raw/832758308905ee20ba9841fa80c47c77d7e58fda/images/logo.png?raw=true" width="100" title="Logo">
</p>

# [Lab 3.0](https://github.com/vpj/lab)

This library helps you organize and track
 machine learning experiments.
 
# [Slack workspace for discussions](https://join.slack.com/t/labforml/shared_invite/zt-cg5iui5u-4cJPT7DUwRGqup9z8RHwhQ)	

[🎛 Dashboard](https://github.com/vpj/lab_dashboard) is the web 
 interface for Lab.


## Features

Main features of Lab are:
* Organizing experiments
* [🎛 Dashboard](https://github.com/vpj/lab_dashboard) to browse experiments
* Logger
* Managing configurations and hyper-parameters

### Organizing Experiments

Lab keeps track of all the model training statistics.
It keeps them in a SQLite database and also pushes them to Tensorboard.
It also organizes the checkpoints and any other artifacts you wish to save.
All of these could be accessed with the Python API and also stored
 in a human friendly folder structure.
This could be  of your training pro
Maintains logs, summaries and checkpoints of all the experiment runs 
 in a folder structure.

### [🎛 Dashboard](https://github.com/vpj/lab_dashboard) to browse experiments
<p align="center">
  <img style="max-width:100%;"
   src="https://raw.githubusercontent.com/vpj/lab/master/images/dashboard.png"
   width="1024" title="Dashboard Screenshot">
</p>

The web dashboard helps navigate experiments and multiple runs.
You can checkout the configs and a summary of performance.
You can launch TensorBoard directly from there.

*Eventually, we want to let you edit configs and run new experiments and analyse
outputs on the dashboard.*

### Logger

Logger has a simple API to produce pretty console outputs.

It also comes with a bunch of helper functions that manages
 iterators and loops.
 
<p align="center">
 <img style="max-width:100%" 
   src="https://raw.githubusercontent.com/vpj/lab/master/images/loop.gif"
  />
</p>

### Manage configurations and hyper-parameters

You can setup configs/hyper-parameters with functions.
[🧪lab](https://github.com/vpj/lab) would identify the dependencies and run 
them in topological order.

```python
@Configs.calc()
def model(c: Configs):
    return Net().to(c.device)
```

You can setup multiple options for configuration functions. 
So you don't have to write a bunch if statements to handle configs.

```python
@Configs.calc(Configs.optimizer)
def sgd(c: Configs):
    return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)

@Configs.calc(Configs.optimizer)
def adam(c: Configs):
    return optim.Adam(c.model.parameters())
```

## Getting Started

### Clone and install

```bash
git clone git@github.com:vpj/lab.git
cd lab
pip install -e .
```

To update run a git update

```bash
cd lab
git pull
```

<!-- ### Install it via `pip` directly from github.

```bash
pip install -e git+git@github.com:vpj/lab.git#egg=lab
``` -->

### Create a `.lab.yaml` file.
An empty file at the root of the project should
be enough. You can set project level configs for
 'check_repo_dirty' and 'path'
in the config file.

Lab will store all experiment data in folder `logs/` 
relative to `.lab.yaml` file.
If `path` is set in `.lab.yaml` then it will be stored in `[path]logs/`
 relative to `.lab.yaml` file.

You don't need the `.lab.yaml` file if you only plan on using the logger.

### [Samples](https://github.com/vpj/lab/tree/master/samples)

[Samples folder](https://github.com/vpj/lab/tree/master/samples) contains a
 bunch of examples of using 🧪 lab.

Here are some [annotated samples](http://blog.varunajayasiri.com/ml/lab3/#samples%2Fmnist_loop.py).

## Tutorial

*The outputs lose color when viewing on github. Run [readme.ipynb](https://github.com/vpj/lab/blob/master/readme.ipynb) locally to try it out.*

This short tutorial covers most of the usage patterns. We still don't have a proper documentation, but the source code of the project is quite clean and I assume you can dive into it if you need more details.

### Logger


```python
from lab import logger
from lab.logger.colors import Text, Color
```

#### Logging with colors


```python
logger.log("Colors are missing when views on github", Text.highlight)
```


<pre><strong><span style="color: #DDB62B">Colors are missing when views on github</span></strong></pre>


You can use predifined styles


```python
logger.log([
    ('Styles ', Text.heading),
    ('Danger ', Text.danger),
    ('Warning ', Text.warning),
    ('Meta ', Text.meta),
    ('Key ', Text.key),
    ('Meta2 ', Text.meta2),
    ('Title ', Text.title),
    ('Heading ', Text.heading),
    ('Value ', Text.value),
    ('Highlight ', Text.highlight),
    ('Subtle', Text.subtle)
])
```


<pre><span style="text-decoration: underline">Styles </span><span style="color: #E75C58">Danger </span><span style="color: #DDB62B">Warning </span><span style="color: #208FFB">Meta </span><span style="color: #60C6C8">Key </span><span style="color: #D160C4">Meta2 </span><strong><span style="text-decoration: underline">Title </span></strong><span style="text-decoration: underline">Heading </span><strong>Value </strong><strong><span style="color: #DDB62B">Highlight </span></strong><span style="color: #C5C1B4">Subtle</span></pre>


Or, specify colors


```python
logger.log([
    ('Colors ', Text.heading),
    ('Red ', Color.red),
    ('Black ', Color.black),
    ('Blue ', Color.blue),
    ('Cyan ', Color.cyan),
    ('Green ', Color.green),
    ('Orange ', Color.orange),
    ('Purple Heading ', [Color.purple, Text.heading]),
    ('White', Color.white),
])
```


<pre><span style="text-decoration: underline">Colors </span><span style="color: #E75C58">Red </span><span style="color: #3E424D">Black </span><span style="color: #208FFB">Blue </span><span style="color: #60C6C8">Cyan </span><span style="color: #00A250">Green </span><span style="color: #DDB62B">Orange </span><span style="color: #D160C4"><span style="text-decoration: underline">Purple Heading </span></span><span style="color: #C5C1B4">White</span></pre>


##### Logging debug info

You can pretty print python objects


```python
logger.info(a=2, b=1)
```


<pre><span style="color: #60C6C8">a: </span><strong>2</strong>
<span style="color: #60C6C8">b: </span><strong>1</strong></pre>



```python
logger.info(dict(name='Name', price=22))
```


<pre><span style="color: #60C6C8"> name: </span><strong>Name</strong>
<span style="color: #60C6C8">price: </span><strong>22</strong>
Total <span style="color: #208FFB">2</span> item(s)</pre>


### Sections

Sections let you monitor time taken for
different tasks and also helps *keep the code clean*
by separating different blocks of code.


```python
import time
```


```python
with logger.section("Load data"):
    # code to load data
    time.sleep(2)
```


<pre>Load data<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	2,005.04ms</span>
</pre>



```python
with logger.section("Load saved model"):
    time.sleep(1)
    logger.set_successful(False)
```


<pre>Load saved model<span style="color: #E75C58">...[FAIL]</span><span style="color: #208FFB">	1,008.86ms</span>
</pre>


You can also show progress while a section is running


```python
with logger.section("Train", total_steps=100):
    for i in range(100):
        time.sleep(0.1)
        # Multiple training steps in the inner loop
        logger.progress(i)
```


<pre>Train<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	10,600.04ms</span>
</pre>


### Iterator and Enumerator

You can use `logger.iterate` and `logger.enumerate` with any iterable object.
In this example we use a PyTorch `DataLoader`.


```python
# Create a data loader for illustration
import torch
from torchvision import datasets, transforms

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True)
```


```python
for data, target in logger.iterate("Test", test_loader):
    time.sleep(0.01)
```


<pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	6,618.08ms</span>
</pre>



```python
for i, (data, target) in logger.enum("Test", test_loader):
    time.sleep(0.01)
```


<pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	6,662.78ms</span>
</pre>


### Loop

This can be used for the training loop.
The `loop` keeps track of the time taken and time remaining for the loop.
You can use *sections*, *iterators* and *enumerators* within loop.

`logger.write` outputs the current status along with global step.


```python
for step in logger.loop(range(0, 400)):
	logger.write()
```


<pre><strong><span style="color: #DDB62B">     399:  </span></strong>  <span style="color: #208FFB">1ms</span><span style="color: #D160C4">  0:00m/  0:00m  </span></pre>


The global step is used for logging to the screen, TensorBoard and when logging analytics to SQLite. You can manually set the global step. Here we will reset it.


```python
logger.set_global_step(0)
```

You can manually increment global step too.


```python
for step in logger.loop(range(0, 400)):
    logger.add_global_step(5)
    logger.write()
```


<pre><strong><span style="color: #DDB62B">   2,000:  </span></strong>  <span style="color: #208FFB">1ms</span><span style="color: #D160C4">  0:00m/  0:00m  </span></pre>


### Log indicators

Here you specify indicators and the logger stores them temporarily and write in batches.
It can aggregate and write them as means or histograms.


```python
import numpy as np

# dummy train function
def train():
    return np.random.randint(100)

# Reset global step because we incremented in previous loop
logger.set_global_step(0)
```

This stores all the loss values and writes the logs the mean on every tenth iteration.
Console output line is replaced until `new_line` is called.


```python
for i in range(1, 401):
    logger.add_global_step()
    loss = train()
    logger.store(loss=loss)
    if i % 10 == 0:
        logger.write()
    if i % 100 == 0:
        logger.new_line()
    time.sleep(0.02)
```


<pre><strong><span style="color: #DDB62B">     100:  </span></strong> loss: <strong> 42.8000</strong>
<strong><span style="color: #DDB62B">     200:  </span></strong> loss: <strong> 39.0000</strong>
<strong><span style="color: #DDB62B">     300:  </span></strong> loss: <strong> 39.7000</strong>
<strong><span style="color: #DDB62B">     400:  </span></strong> loss: <strong> 53.6000</strong></pre>


#### Indicator settings


```python
from lab.logger.indicators import Queue, Scalar, Histogram

# dummy train function
def train2(idx):
    return idx, 10, np.random.randint(100)

# Reset global step because we incremented in previous loop
logger.set_global_step(0)
```

`Histogram` indicators will log a histogram of data.
`Queue` will store data in a `deque` of size `queue_size`, and log histograms.
Both of these will log the means too. And if `is_print` is `True` it will print the mean.

queue size of `10` and the values are printed to the console


```python
logger.add_indicator(Queue('reward', 10, True))
```

By default values are not printed to console; i.e. `is_print` defaults to `False`.


```python
logger.add_indicator(Scalar('policy'))
```

Settings `is_print` to `True` will print the mean value of histogram to console


```python
logger.add_indicator(Histogram('value', True))
```


```python
for i in range(1, 400):
    logger.add_global_step()
    reward, policy, value = train2(i)
    logger.store(reward=reward, policy=policy, value=value, loss=1.)
    if i % 10 == 0:
        logger.write()
    if i % 100 == 0:
        logger.new_line()
```


<pre><strong><span style="color: #DDB62B">     100:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 95.5000</strong> value: <strong> 40.5000</strong>
<strong><span style="color: #DDB62B">     200:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 195.500</strong> value: <strong> 53.4000</strong>
<strong><span style="color: #DDB62B">     300:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 295.500</strong> value: <strong> 60.3000</strong>
<strong><span style="color: #DDB62B">     390:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 385.500</strong> value: <strong> 45.0000</strong></pre>


### Experiment

Lab will keep track of experiments if you declare an Experiment. It will keep track of logs, code diffs, git commits, etc.


```python
from lab.experiment.pytorch import Experiment
```

The `name` of the defaults to the calling python filename. However when invoking from a Jupyter Notebook it must be provided because the library cannot find the calling file name. `comment` can be changed later from the [🎛 Dashboard](https://github.com/vpj/lab_dashboard).


```python
exp = Experiment(name="mnist_pytorch",
                 comment="Test")
```

Starting an experiments creates folders, stores the experiment meta data, git commits, and source diffs.


```python
exp.start()
```


<pre><strong><span style="text-decoration: underline">mnist_pytorch</span></strong>: <span style="color: #208FFB">a4a7586664f411eab095acde48001122</span>
	<strong><span style="color: #DDB62B">Test</span></strong>
	[dirty]: <strong><span style="color: #DDB62B">"Update readme.md"</span></strong></pre>


You can also start from a previously saved checkpoint. A `run_uuid` of `` means that it will load from the last run.

```python
exp.start(run_uuid='')
```

### Configs


```python
from lab import configs
```

The configs will be stored and in future be adjusted from  [🎛 Dashboard](https://github.com/vpj/lab_dashboard)


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

