<p align="center">
  <img src="https://github.com/vpj/lab/raw/832758308905ee20ba9841fa80c47c77d7e58fda/images/logo.png?raw=true" width="100" title="Logo">
</p>

# [🚧 Lab 3.0](https://github.com/vpj/lab)

**⚠️ This is the development version, with a lot of frequent bareaking changes.**
Feel free to try it out.
Use the [previous](https://github.com/vpj/lab/tree/2.0) stable version if you want to use it in a project.

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

The web dashboard helps navigate experiments and multiple runs.
You can checkout the configs and a summary of performance.
You can launch TensorBoard directly from there.

Eventually, we want to let you edit configs and run new experiments and analyse
outputs on the dashboard.

### Logger

Logger has a simple API to produce pretty console outputs.

<p align="center"><img style="max-width:100%" src="https://github.com/vpj/lab/raw/832758308905ee20ba9841fa80c47c77d7e58fda/images/loop.gif" /></p>

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

#### Install it via `pip` directly from github.

```bash
pip install -e git+git@github.com:vpj/lab.git#egg=lab
```

#### Create a `.lab.yaml` file.
An empty file at the root of the project should
be enough. You can set project level configs for
 'check_repo_dirty' and 'path'
in the config file.

Lab will store all experiment data in folder `logs/` 
relative to `.lab.yaml` file.
If `path` is set in `.lab.yaml` then it will be stored in `[path]logs/`
 relative to `.lab.yaml` file.

You don't need the `.lab.yaml` file if you only plan on using the logger.

### Samples

[Samples folder](https://github.com/vpj/lab/tree/master/samples) contains a
 bunch of examples of using 🧪 lab.

## Usage Example


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
logger.log("Red", color=Text.danger)
logger.log([
    ('Red', Color.red),
    ' Normal ',
    ('Cyan', Color.cyan)
])
```

    [31mRed[0m
    [31mRed[0m Normal [36mCyan[0m


##### Logging debug info


```python
logger.info(a=2, b=1)
logger.info(dict(name='Name', price=22))
```

    [36ma: [0m[1m2[0m
    [36mb: [0m[1m1[0m
    [36m name: [0m[1mName[0m
    [36mprice: [0m[1m22[0m
    Total [34m2[0m item(s)


### Log indicators

Here you specify indicators and the logger stores them temporarily and write in batches.
It can aggregate and write them as means or histograms.


```python
# dummy train function
def train():
    return np.random.randint(100)
```

This stores all the loss values and writes the logs the mean on every tenth iteration.
Console output line is replaced until `new_line` is called.


```python
for i in range(1, 1000):
    logger.add_global_step()
    loss = train()
    logger.store(loss=loss)
    if i % 10 == 0:
        logger.write()
    if i % 100 == 0:
        logger.new_line()
```

    [1m[33m     100:  [0m loss: [1m 52.2000[0m
    [1m[33m     200:  [0m loss: [1m 61.7000[0m
    [1m[33m     300:  [0m loss: [1m 62.8000[0m
    [1m[33m     400:  [0m loss: [1m 53.0000[0m
    [1m[33m     500:  [0m loss: [1m 51.0000[0m
    [1m[33m     600:  [0m loss: [1m 36.2000[0m
    [1m[33m     700:  [0m loss: [1m 51.5000[0m
    [1m[33m     800:  [0m loss: [1m 49.5000[0m
    [1m[33m     900:  [0m loss: [1m 49.4000[0m
    [1m[33m     990:  [0m loss: [1m 56.4000[0m

#### Indicator settings


```python
from lab.logger.indicators import Queue, Scalar, Histogram
```


```python
# dummy train function
def train2(idx):
    return idx, 10, np.random.randint(100)
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
logger.set_global_step(0)
for i in range(1, 1000):
    logger.add_global_step()
    reward, policy, value = train2(i)
    logger.store(reward=reward, policy=policy, value=value, loss=1.)
    if i % 10 == 0:
        logger.write()
    if i % 100 == 0:
        logger.new_line()
```

    [1m[33m     100:  [0m loss: [1m 1.00000[0m reward: [1m  95.500[0m value: [1m 41.1000[0m
    [1m[33m     200:  [0m loss: [1m 1.00000[0m reward: [1m 195.500[0m value: [1m 60.8000[0m
    [1m[33m     300:  [0m loss: [1m 1.00000[0m reward: [1m 295.500[0m value: [1m 62.2000[0m
    [1m[33m     400:  [0m loss: [1m 1.00000[0m reward: [1m 395.500[0m value: [1m 25.6000[0m
    [1m[33m     500:  [0m loss: [1m 1.00000[0m reward: [1m 495.500[0m value: [1m 62.3000[0m
    [1m[33m     600:  [0m loss: [1m 1.00000[0m reward: [1m 595.500[0m value: [1m 52.3000[0m
    [1m[33m     700:  [0m loss: [1m 1.00000[0m reward: [1m 695.500[0m value: [1m 54.5000[0m
    [1m[33m     800:  [0m loss: [1m 1.00000[0m reward: [1m 795.500[0m value: [1m 36.8000[0m
    [1m[33m     900:  [0m loss: [1m 1.00000[0m reward: [1m 895.500[0m value: [1m 55.4000[0m
    [1m[33m     990:  [0m loss: [1m 1.00000[0m reward: [1m 985.500[0m value: [1m 51.6000[0m

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

    Load data[32m...[DONE][0m[34m	2,001.41ms[0m
    Load saved model[31m...[FAIL][0m[34m	1,006.21ms[0m


#### Progress

This shows the progress for code within the section.


```python
with logger.section("Train", total_steps=100):
    for i in range(100):
        time.sleep(0.1)
        # Multiple training steps in the inner loop
        logger.progress(i)
```

    Train[32m...[DONE][0m[34m	10,457.20ms[0m


##### Iterator and Enumerator
```python
for data, target in logger.iterator("Test", test_loader):
    ...

for i, (data, target) in logger.enumerator("Train", train_loader):
    ...
```

This combines `section` and `progress`

### Loop

```python
for step in logger.loop(range(0, total_steps)):
	# training code ...
```

The `loop` keeps track of the time taken and time remaining for the loop.
You can use *sections*, *iterators* and *enumerators* within loop.


### Experiment

```python
exp = Experiment(name="mnist_pytorch",
                 comment="Test")
```

* `name`: Name of the experiment;
 If not provided, this defaults to the calling python filename
* `comment`: Comment about the current experiment run

##### Starting an expriemnt
```python
exp.start()
```

This does the initialization work like creating log folders, writing run details, etc.

```python
exp.start(run_index=-1)
```

This starts the experiment from where it was left in the last run.
You can provide a specific the run index to start from.

##### Save Checkpoint
```python
logger.save_checkpoint()
```

This saves a checkpoint and you can start from the saved checkpoint with
`exp.start(run=-1)`.

### 🚧 Configs
### 🚧 Training Loop

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
