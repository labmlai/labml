<p align="center">
  <img src="https://github.com/vpj/lab/blob/master/images/logo.png?raw=true" width="100" title="Logo">
</p>

# [Lab 3.0](https://github.com/vpj/lab)

This library helps you organize machine learning experiments.
It is a quite small library,
 and most of the modules can be used independently of each other.
This doesn't have any user interface.
Experiment results are maintained in a folder structure,
and there is a Python API to access them.

## Features

### Organize Experiments

Maintains logs, summaries and checkpoints of all the experiment runs in a folder structure.

```
logs
├── experiment1
│   ├── run1
│   │   ├── run.yaml
│   │   ├── configs.yaml
│   │   ├── indicators.yaml
│   │   ├── source.diff
│   │   ├── checkpoints
│   │   │   └── 📄 Saved checkpoints
│   │   └── tensorboard
│   │       └── 📄 TensorBoard summaries
│   ├── run1
│   ...
└── experiment2...
    ├──
    ...
```

### [🎛 Dashboard](https://github.com/vpj/lab_dashboard) to browse experiments

<p align="center">
  <img style="max-width:100%;" src="https://github.com/vpj/lab/blob/master/images/dashboard.png?raw=true" width="320" title="Logo">
</p>

The web dashboard helps navigate experiments and multiple runs.
You can checkout the configs and a summary of performance.
You can launch TensorBoard directly from there.

Eventually, we want to let you edit configs and run new experiments and analyse
outputs on the dashboard.

### Logger

Logger has a simple API to produce pretty console outputs.

<p align="center"><img style="max-width:100%" src="https://github.com/vpj/lab/blob/master/images/loop.gif" /></p>

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
@Configs.calc('optimizer')
def sgd(c: Configs):
    return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)

@Configs.calc('optimizer')
def adam(c: Configs):
    return optim.Adam(c.model.parameters())
```

## Getting Started

### Install it via `pip` directly from github.

```bash
pip install -e git+git@github.com:vpj/lab.git#egg=lab
```

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

## Tutorial

*The outputs lose color when viewing on github. Run [readme.ipynb](https://github.com/vpj/lab/blob/master/readme.ipynb) locally to try it out.*

This short tutorial covers most of the usage patterns. We still don't have a proper documentation, but the source code of the project is quite clean and I assume you can dive into it if you need more details.


```python
# Some imports
import numpy as np
import time
```

### Logger


```python
from lab import logger
from lab.logger.colors import Text, Color
```

#### Logging with colors


```python
logger.log("Colors are missing when views on github", Text.highlight)

logger.log([
    ('Styles\n', Text.heading),
    ('Danger\n', Text.danger),
    ('Warning\n', Text.warning),
    ('Meta\n', Text.meta),
    ('Key\n', Text.key),
    ('Meta2\n', Text.meta2),
    ('Title\n', Text.title),
    ('Heading\n', Text.heading),
    ('Value\n', Text.value),
    ('Highlight\n', Text.highlight),
    ('Subtle\n', Text.subtle)
])

logger.log([
    ('Colors\n', Text.heading),
    ('Red\n', Color.red),
    ('Black\n', Color.black),
    ('Blue\n', Color.blue),
    ('Cyan\n', Color.cyan),
    ('Green\n', Color.green),
    ('Orange\n', Color.orange),
    ('Purple Heading\n', [Color.purple, Text.heading]),
    ('White\n', Color.white),
])
```


<pre><strong><span style="color: #DDB62B">Colors are missing when views on github</span></strong>
<span style="text-decoration: underline">Styles</span>
<span style="text-decoration: underline"></span><span style="color: #E75C58">Danger</span>
<span style="color: #E75C58"></span><span style="color: #DDB62B">Warning</span>
<span style="color: #DDB62B"></span><span style="color: #208FFB">Meta</span>
<span style="color: #208FFB"></span><span style="color: #60C6C8">Key</span>
<span style="color: #60C6C8"></span><span style="color: #D160C4">Meta2</span>
<span style="color: #D160C4"></span><strong><span style="text-decoration: underline">Title</span></strong>
<strong><span style="text-decoration: underline"></span></strong><span style="text-decoration: underline">Heading</span>
<span style="text-decoration: underline"></span><strong>Value</strong>
<strong></strong><strong><span style="color: #DDB62B">Highlight</span></strong>
<strong><span style="color: #DDB62B"></span></strong><span style="color: #C5C1B4">Subtle</span>
<span style="color: #C5C1B4"></span>
<span style="text-decoration: underline">Colors</span>
<span style="text-decoration: underline"></span><span style="color: #E75C58">Red</span>
<span style="color: #E75C58"></span><span style="color: #3E424D">Black</span>
<span style="color: #3E424D"></span><span style="color: #208FFB">Blue</span>
<span style="color: #208FFB"></span><span style="color: #60C6C8">Cyan</span>
<span style="color: #60C6C8"></span><span style="color: #00A250">Green</span>
<span style="color: #00A250"></span><span style="color: #DDB62B">Orange</span>
<span style="color: #DDB62B"></span><span style="color: #D160C4"><span style="text-decoration: underline">Purple Heading</span></span>
<span style="color: #D160C4"><span style="text-decoration: underline"></span></span><span style="color: #C5C1B4">White</span>
<span style="color: #C5C1B4"></span></pre>


##### Logging debug info


```python
logger.info(a=2, b=1)
logger.info(dict(name='Name', price=22))
```


<pre><span style="color: #60C6C8">a: </span><strong>2</strong>
<span style="color: #60C6C8">b: </span><strong>1</strong>
<span style="color: #60C6C8"> name: </span><strong>Name</strong>
<span style="color: #60C6C8">price: </span><strong>22</strong>
Total <span style="color: #208FFB">2</span> item(s)</pre>


### Sections

Sections let you monitor time taken for
different tasks and also helps *keep the code clean*
by separating different blocks of code.


```python
with logger.section("Load data"):
    # code to load data
    time.sleep(2)

with logger.section("Load saved model"):
    time.sleep(1)
    logger.set_successful(False)
    # code to create model
```


<pre>Load data<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	2,007.24ms</span>
Load saved model<span style="color: #E75C58">...[FAIL]</span><span style="color: #208FFB">	1,008.62ms</span>
</pre>


#### Progress

This shows the progress for code within the section.


```python
with logger.section("Train", total_steps=100):
    for i in range(100):
        time.sleep(0.1)
        # Multiple training steps in the inner loop
        logger.progress(i)
```


<pre>Train<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	10,641.20ms</span>
</pre>


### Iterator and Enumerator

This combines `section` and `progress`. In this example we use a PyTorch `DataLoader`. You can use `logger.iterate` and `logger.enumerate` with any iterable object.


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


<pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	6,726.97ms</span>
</pre>



```python
for i, (data, target) in logger.enum("Test", test_loader):
    time.sleep(0.01)
```


<pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	6,816.58ms</span>
</pre>


### Loop

The `loop` keeps track of the time taken and time remaining for the loop.
You can use *sections*, *iterators* and *enumerators* within loop.

`logger.write` outputs the current status along with global step.


```python
for step in logger.loop(range(0, 400)):
	logger.write()
```


<pre><strong><span style="color: #DDB62B">     400:  </span></strong>  <span style="color: #208FFB">1ms</span><span style="color: #D160C4">  0:00m/ -1:59m  </span></pre>


#### Global step

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


<pre><strong><span style="color: #DDB62B">     100:  </span></strong> loss: <strong> 59.7000</strong>
<strong><span style="color: #DDB62B">     200:  </span></strong> loss: <strong> 47.2000</strong>
<strong><span style="color: #DDB62B">     300:  </span></strong> loss: <strong> 43.3000</strong>
<strong><span style="color: #DDB62B">     400:  </span></strong> loss: <strong> 46.0000</strong></pre>


#### Indicator settings


```python
from lab.logger.indicators import Queue, Scalar, Histogram
```


```python
# dummy train function
def train2(idx):
    return idx, 10, np.random.randint(100)

# Reset global step because we incremented in previous loop
logger.set_global_step(0)
```

`histogram` indicators will log a histogram of data.
`queue` will store data in a `deque` of size `queue_size`, and log histograms.
Both of these will log the means too. And if `is_print` is `True` it will print the mean.


```python
# queue_size = 10,
logger.add_indicator(Queue('reward', 10, True))
# is_print default to False
logger.add_indicator(Scalar('policy'))
# is_print = True
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


<pre><strong><span style="color: #DDB62B">     100:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 95.5000</strong> value: <strong> 42.2000</strong>
<strong><span style="color: #DDB62B">     200:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 195.500</strong> value: <strong> 53.2000</strong>
<strong><span style="color: #DDB62B">     300:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 295.500</strong> value: <strong> 47.9000</strong>
<strong><span style="color: #DDB62B">     390:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 385.500</strong> value: <strong> 45.1000</strong></pre>


### Experiment

Lab will keep track of experiments if you declare an Experiment. It will keep track of logs, code diffs, git commits, etc.


```python
from lab.experiment.pytorch import Experiment
```


```python
exp = Experiment(name="mnist_pytorch",
                 comment="Test")
```

The `name` of the defaults to the calling python filename. However when invoking from a Jupyter Notebook it must be provided because the library cannot find the calling file name.

#### Starting an expriemnt

This creates the folders, stores the experiment meta data, git commits, and source diffs.


```python
exp.start()
```


<pre><strong><span style="text-decoration: underline">mnist_pytorch</span></strong>: <span style="color: #208FFB">1</span>
	<strong><span style="color: #DDB62B">Test</span></strong>
	[dirty]: <strong><span style="color: #DDB62B">"🧹  readme"</span></strong></pre>


You can also start from a previously saved checkpoint. A `run_index` of `-1` means that it will load from the last run.

```python
exp.start(run_index=-1)
```

#### Save Checkpoint


```python
logger.save_checkpoint()
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
@DeviceConfigs.calc('device')
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

Configs classes can be inherited


```python
class Configs(DeviceConfigs):
    model_size: int = 10

    model: any = 'cnn_model'
```

You can specify multiple config calculator functions. The function given by the string for respective attribute will be picked.


```python
@Configs.calc('model')
def cnn_model(c: Configs):
    return c.model_size * 10

@Configs.calc('model')
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


<pre><span style="color: #60C6C8">model: </span><strong>20</strong></pre>


---

## Background
I was coding existing reinforcement learning algorithms
 to play Atari games for fun.
It was not easy to keep track of things when I started
 trying variations, fixing bugs etc.
Then I wrote some tools to organize my experiment runs.
I found it important to keep track of git commits
to make sure I can reproduce results.

I also wrote a logger to display pretty results on screen and
 to make it easy to write TensorBoard summaries.
It also keeps track of training times which makes it easy to spot
 what's taking up most resources.

This library is was made by combining these bunch of tools.

## Alternatives

### Managing Experiments

* [Comet](https://www.comet.ml/)
* [Beaker](https://beaker.org/)
* [Sacred](https://github.com/IDSIA/sacred)
* [Neptune](https://neptune.ml/)
* [Model Chimp](https://www.modelchimp.com/)

### Logging

* [TQDM](https://tqdm.github.io/)
* [Loguru](https://github.com/Delgan/loguru)


```python

```
