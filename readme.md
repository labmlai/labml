<p align="center">
  <img src="/images/logo.png?raw=true" width="100" title="Logo">
</p>

# [Lab 2.0](https://github.com/vpj/lab)

This library helps you organize machine learning experiments.
It is a quite small library, 
 and most of the modules can be used independently of each other.
This doesn't have any user interface.
Experiment results are maintained in a folder structure,
and there is a Python API to access them.

## Features

### Organize Experiments

Maintains logs, summaries and checkpoints of all the experiments in a folder
structure without you explicitly having to worry about them.

```
logs
├── experiment1
│   ├── checkpoints
│   │   └── 📄 Saved checkpoints
│   ├── log
│   │   └── 📄 TensorBoard summaries
│   ├── diffs
│   │   └── 📄 diffs when experiment was run
│   └── trials.yaml
└── experimnent2...
    ├──
    ...
```

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
  - ...
  python_file: /progrect/experiments/mnist_simple_convolution.py
  start_step: '0'
  trial_date: '2019-06-14'
  trial_time: '16:13:18'
- ...
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
|             ...                   |
-------------------------------------
"""
```

### Custom Visualizations of TensorBoard summaries

<p align="center"><img style="max-width:100%" src="/images/distribution.png" /></p>

With the visualization helper functions you can plot distributions easily on a Jupyter Notebook.

Here's a link to a
 [sample Jupyter Notebook with custom charts](https://github.com/vpj/lab/blob/master/analytics/sample_analytics.ipynb).

### Logger

Logger has a simple API to produce pretty console outputs.

<p align="center"><img style="max-width:100%" src="/images/loop.gif" /></p>

---

## Getting Started

Clone this repository and add a symbolic link to lab.

```bash
ln -s ~/repo/lab your_project/lab
```

The create a `.lab.yaml` file. An empty file at the root of the project should
be enough. You can set project level configs for 'check_repo_dirty' and 'is_log_python_file'
in the config file.

The idea is to have a separate python file for each major expirment,
 like different architectures.
Minor changes can go as trials, like bug fixes and improvements.
The TensorBoard summaries are replaced for each trial.

You don't need the `.lab.yaml` file if you only plan on using the logger.

### Samples

See [mnist_pytorch.py](http://blog.varunajayasiri.com/ml/lab/mnist_pytorch.html) 
or [mnist_tensorflow.py](http://blog.varunajayasiri.com/ml/lab/mnist_tensorflow.html)
for examples.

## Usage

### Create Experiment
```python
EXPERIMENT = Experiment(name="mnist_pytorch",
                        python_file=__file__,
                        comment="Test",
                        check_repo_dirty=False,
			is_log_python_file=True)
```

* `name`: Name of the experiment
* `python_file`: The python file with the experiment definition.
* `comment`: Comment about the current experiment trial
* `check_repo_dirty`: If `True` the experiment is halted if there are uncommitted changes to the git repository.
* `is_log_python_file`: Whether to update the python source file with experiemnt results on the top.

```python
EXPERIMENT.start_train()
```

You need to call `start_train` before starting the experiment to clear old logs and
do other initialization work.

It will load from a saved state if you call `EXPERIMENT.start_train(False)`.

Call `start_replay`, when you want to just evaluate a model by loading from saved checkpoint.

```python
EXPERIMENT.start_replay()
```

### Logger

`EXPERIMENT.logger` gives logger instance for the experiment.

You can also directly initialize a logger with `Logger()`,
in which case it will only output to the screen.

### Loop

```python
for step in logger.loop(range(0, total_steps)):
	# training code ...
```

The `Loop` keeps track of the time taken and time remaining for the loop.

### Sections

```python
with logger.section("Load data"):
    # code to load data
with logger.section("Create model"):
    # code to create model
```

Sections let you monitor time takes for
different tasks and also helps keep the code clean 
by separating different blocks of code.

These can be within loops as well.

### Progress

```python
for step in logger.loop(range(0, total_steps)):
	with logger.section("train", total_steps=100):
	    for i in range(100):
		# Multiple training steps in the inner loop
		logger.progress(i)
	    # Clears the progress when complete
```

This shows the progress for training in the inner loop.
You can do progress monitoring within sections outside the
 `logger.loop` as well.

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


### Save Progress
```python
logger.save_progress()
```

This saves the progress stats in `trials.yaml` and python file header

### Save Checkpoint
```python
logger.save_checkpoint()
```

This saves a checkpoint and you can start from the saved checkpoint with
`EXPERIMENT.start_train(global_step)`, or with `EXPERIMENT.start_replay(global_step)`
if you just want inference. When started with `start_replay` it won't update any of the logs.

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

### Start TensorBoard
This small tool lets you start TensorBoard without having to type in all the log paths.

To get a list of all available experiments
```bash
LAB/tb.py -l
```

To analyse experiments `exp1` and `exp2`:

```bash
LAB/tb.py -e exp1 exp2
```

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

* **July 12, 2019**
	* TensborBoard embedding helper
	
* **July 15, 2019**
	* Nested sections
	* Helpers for iterators
	
* **July 17, 2019**
	* Singleton logger - 
	  *much simpler usage although it might not be able to handle complexities later*
