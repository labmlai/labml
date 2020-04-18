<p align="center">
  <img src="https://github.com/vpj/lab/raw/832758308905ee20ba9841fa80c47c77d7e58fda/images/logo.png?raw=true" width="100" title="Logo">
</p>

# [Lab 3.0](https://github.com/vpj/lab)

This library helps you organize and track
 machine learning experiments.
 
**💬 [Slack workspace for discussions](https://join.slack.com/t/labforml/shared_invite/zt-cg5iui5u-4cJPT7DUwRGqup9z8RHwhQ)**	

**🎛 [Dashboard](https://github.com/vpj/lab_dashboard) is the web 
 interface for Lab.**


## Features

Main features of Lab are:
* Organizing experiments
* [Dashboard](https://github.com/vpj/lab_dashboard) to browse experiments
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

### [Dashboard](https://github.com/vpj/lab_dashboard) to browse experiments
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
[Lab](https://github.com/vpj/lab) would identify the dependencies and run 
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