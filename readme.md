# 🧪 Lab 2.0

[Github Repository](https://github.com/vpj/lab)

This library lets you organize machine learning
 experiments.

## Features

### Organize checkpoints, TensorBoard summaries and logs
Maintains logs, summaries and checkpoints of all the experiments in a folder
structure without you explicitly having to worry about them.

```
logs
├── mnist_convolution
│   ├── log
│   │   └── 📄 TensorBoard summaries
│   └── trials.yaml
└── mnist_attention
    ├── checkpoints
    │   └── 📄 Saved checkpoints
    ├── log
    │   └── 📄 TensorBoard summaries
    └── trials.yaml
```

### Keep track of experiments
The `trials.yaml` file keeps the summaries of each run for that experiment in
a human readable form.

```yaml
- comment: 📊  with gradients
  commit: f763d41b7f5d2ca4dd4431d5a5354d937ed3d32d
  commit_message: "📊 logging gradient summaries"
  is_dirty: 'true'
  progress:
  - accuracy: '    0.30'
    global_step: '   3,752'
    test_loss: '    2.22'
    train_loss: '    2.22'
  - accuracy: '    0.71'
    global_step: '  11,256'
    test_loss: '    1.03'
    train_loss: '    1.06'
  - accuracy: '    0.78'
    global_step: '  18,760'
    test_loss: '    0.63'
    train_loss: '    0.69'
  - accuracy: '    0.83'
    global_step: '  26,264'
    test_loss: '    0.49'
    train_loss: '    0.53'
  - accuracy: '    0.87'
    global_step: '  34,706'
    test_loss: '    0.38'
    train_loss: '    0.39'
  python_file: /progrect/experiments/mnist_simple_convolution.py
  start_step: '0'
  trial_date: '2019-06-14'
  trial_time: '16:13:18'
- comment: 🐛 stride fix
...
```

It keeps references to **git commit** when the experiement was run,
along with other information like date, the python file executed and
experiment description.

Optionally, the library can update the python file by
 inserting experiment results as a comment 👇 automatically.

```python
"""
```trial
2019-02-12 16:03:16
Sample lab experiment
[[dirty]]: 🤪 jupyter saved
start_step: 0

-------------------------------------
| global_step |   reward |     loss |
-------------------------------------
|           9 |     1.50 |    13.50 |
|          19 |     4.83 |    23.50 |
|          29 |     8.17 |    33.50 |
|          39 |    11.50 |    43.50 |
|          49 |    14.83 |    53.50 |
-------------------------------------
"""
```

### Timing and progress
You can use monitored code segments to measure time 
and to get status updates on the console.
This also helps organize the code.

```python
with logger.monitor("Load data"):
    # code to load data
with logger.monitor("Create model"):
    # code to create model
```

will produce an output like
<p align="center"><img style="max-width:100%" src="http://blog.varunajayasiri.com/ml/lab/images/monitored_sections.png" /></p>

Library also has utility functions to monitor loops.
<p align="center"><img style="max-width:100%" src="http://blog.varunajayasiri.com/ml/lab/images/loop.gif" /></p>

### Customized Visualizations of TensorBoard summaries

TensorBoard is nice, but sometimes you need
custom charts to debug algorithms. Following
is an example of a custom chart:

<p align="center"><img style="max-width:100%" src="http://blog.varunajayasiri.com/ml/lab/images/distribution.png" /></p>

And sometime TensorBoard is not even doing a good job;
for instance lets say you have a histogram, with 90% of
data points between 1 and 2 whilst there are a few outliers at 1000 -
you won't be able to see the distribution between 1 and 2 because
the graph is scaled to 1000.

I think TensorBoard will develop itself to handle these.
And the main reason behind these tooling I've written
is for custom charts, and because it's not that hard to do it.

Here's a link to a
 [sample Jupyter Notebook with custom charts](https://github.com/vpj/lab/blob/master/sample_analytics.ipynb).

### Handle Keyboard Interrupts

You can use this to delay the Keyboard interrupts and make sure
a given segment of code runs without interruptions.

### Start TensorBoard
This lets you start TensorBoard without having to type in all the log paths.
For instance, so that you can start it with
```bash
python tools/tb.py -e ppo ppo_transformed_bellman
```

instead of
```bash
tensorboard --log_dir=ppo:project_path/logs/ppo,ppo_transformed_bellman:project_path/logs/ppo_transformed_bellman
```

To get a list of all available experiments
```bash
python tools/tb.py -e ppo ppo_transformed_bellman
```

* A simple TensorBoard invoker
* Tools for custom graphs from TensorBoard summaries


### Colored console outputs
The logger creates beautiful colorized console outputs that's easy on the eye.

<p align="center"><img style="max-width:100%" src="http://blog.varunajayasiri.com/ml/lab/images/log.png" /></p>

### Histograms and moving averages
The logger can buffer data and produce moving averages and
TensorBoard histograms.
This saves you the extra code to buffering.


## Getting Started

Clone this repository and add a symbolic link to lab.

```bash
ln -s ~/repo/lab your_project/lab
ln -s ~/repo/tools your_project/tools
```

The create a `lab_globals.py` file and set project level configurations.
See [lab_globals.py](http://blog.varunajayasiri.com/ml/lab/lab_globals.html) for example.

### A python file for each experiment
The idea is to have a separate python file for each major expirment,
 like different architectures.
Minor changes can go as trials, like bug fixes and improvements.
The TensorBoard summaries are replaced for each trial.

### Samples

See [mnist_pytorch.py](http://blog.varunajayasiri.com/ml/lab/mnist_pytorch.html) 
or [mnist_tensorflow.py](http://blog.varunajayasiri.com/ml/lab/mnist_tensorflow.html)
for examples.

## Usage

### Monitored Sections

```python
with logger.section("Load data"):
    # code to load data
with logger.section("Create model"):
    # code to create model
```

Monitored sections let you monitor time takes for
different tasks and also helps keep the code clean 
by separating different sections.

### Monitored Iterator

```python
for step in logger.loop(range(0, total_steps)):
	# training code ...
```

The monitored iterator keeps track of the time taken and time remaining for the loop.
`print_global_step` prints the global step to the console.
`monitor.progress()` prints the time taken and time remaining.

#### Progress Monitoring within Loop

```python
for step in logger.loop(range(0, total_steps)):
	with logger.section("train", total_steps=100):
	    for i in range(100):
		# Multiple training steps in the inner loop
		logger.progress(i)
	    # Clears the progress when complete
```

This shows the progress for training in the inner loop.
This behaviour was necessary in Reinforcement Learning where the
 main loop gathers samples and trains;
 whilst the inner sampling and training loops also run for a few steps.

### Log indicators

```python
logger.add_indicator("reward", queue_limit=10)
logger.add_indicator("fps", is_histogram=False, is_progress=False)
logger.add_indicator("loss", is_histogram=True)
logger.add_indicator("advantage_reward", is_histogram=False, is_print=False, is_pair=True)
```

* `queue_limit: int = None`: If the queue size is specified the values are added to a fixed sized queue and the mean and histogram can be used.
* `is_histogram: bool = True`: If true a TensorBoard histogram summaries is produced along with the mean scalar.
* `is_print: bool = True`: If true the mean value is printed to the console
* `is_progress: Optional[bool] = None`: If true the mean value is recorded in experiment summary in `trials.yaml` and in the python file header. If a value is not provided it is set to be equal to `is_print`.
* `is_pair: bool = False`: Whether the values are pairs of values. *This is still experimental*. This can be used to produce multi dimensional visualizations.

The values are stored using `logger.store` function.

```python
logger.store(
    reward=global_step / 3.0,
    fps=12
)
logger.store('loss', i)
logger.store(advantage_reward=(i, i * 2))
```

### Write Logs
```python
logger.write()
```

This will write the stored and values in the logger and clear the buffers.
It will write to console as well as TensorBoard summaries.

In standard usage of this library we do not move to new_lines after each console output.
Instead we update the stats on the same line and move to a new line after a few iterations.

```python
logger.new_line()
```

This will start a new line in the console.

### Create Experiment
```python
EXPERIMENT = Experiment(lab=lab,
                        name="mnist_pytorch",
                        python_file=__file__,
                        comment="Test",
                        check_repo_dirty=False,
			is_log_python_file=True)
```

* `lab`: This is the project level lab definition. See [lab_global.py](http://blog.varunajayasiri.com/ml/lab/lab_global.html) for example.
* `name`: Name of the experiment
* `python_file`: The python file with the experiment definition.
* `comment`: Comment about the current experiment trial
* `check_repo_dirty`: If `True` the experiment is halted if there are uncommitted changes to the git repository.
* `is_log_python_file`: Whether to update the python source file with experiemnt results on the top.

```python
EXPERIMENT.start_train(0)
```

You need to call `start_train` before starting the experiment to clear old logs and
do other initialization work.

It will load from a saved state if the `global_step` is not `0`.
*(🚧 Not implemented for PyTorch yet)*

### Save Progress
```python
EXPERIMENT.save_progress(logger.progress_dict)
```

This saves the progress stats in `trials.yaml` and python file header

### Save Checkpoint
```python
EXPERIMENT.save_checkpoint(global_step)
```

This saves a checkpoint and you can start from the saved checkpoint with
`EXPERIMENT.start_train(global_step)`, or with `EXPERIMENT.start_replay(global_step)`
if you just want inference. When started with `start_replay` it won't update any of the logs.
*(🚧 Not implemented for PyTorch yet)*

### Keyboard Interrupts

```python
try:
    with logger.delayed_keyboard_interrupt():
    	# segment of code that needs to run without breaking in the middle
except KeyboardInterrupt:
	# handle the interruption after the code segment is executed
```

You can wrap a segment of code that needs to run without interruptions
within a `with logger.delayed_keyboard_interrupt()`.

Two consecutive interruptions will halt the code segment.

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

## Updates

* **November 16, 2018**
    * Initial public release

* **December 21, 2018**
    * TensorBoard invoker
    * Tool set for custom visualizations of TensorBoard summaries
    * Automatically adds headers to python files with experiment results

* **February 10, 2019**
	* Two dimensional summaries

* **June 16, 2019**
	* PyTorch support
	* Improved Documentation
	* MNIST examples

* **June 19, 2019**
	* New API for logger

## 🖋 [@vpj on Twitter](https://twitter.com/vpj)
