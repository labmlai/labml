<div align="center" style="margin-bottom: 100px;">

<h1>Monitor deep learning model training and hardware usage from mobile.</h1>

[![PyPI - Python Version](https://badge.fury.io/py/labml.svg)](https://badge.fury.io/py/labml)
[![PyPI Status](https://pepy.tech/badge/labml)](https://pepy.tech/project/labml)
[![Docs](https://img.shields.io/badge/labml-docs-blue)](https://docs.labml.ai/)
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai?ref_src=twsrc%5Etfw)

<img src="https://github.com/labmlai/labml/blob/master/images/cover-dark.png" alt=""/>
</div>

### 🔥 Features

* Monitor running experiments from [mobile phone](https://github.com/labmlai/labml/tree/master/app) (or laptop)
* Monitor [hardware usage on any computer](https://github.com/labmlai/labml/blob/master/guides/hardware_monitoring.md)
  with a single command
* Integrate with just 2 lines of code (see examples below)
* Keeps track of experiments including infomation like git commit, configurations and hyper-parameters
* Keep Tensorboard logs organized
* Save and load checkpoints
* API for custom visualizations
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/labml/blob/master/samples/stocks/analysis.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vpj/poker/blob/master/kuhn_cfr/kuhn_cfr.ipynb)
* Pretty logs of training progress
* [Change hyper-parameters while the model is training](https://github.com/labmlai/labml/blob/master/guides/dynamic_hyperparameters.md)
* Open source! we also have a small hosted server for the mobile web app

### Installation

You can install the packages using PIP.

```bash
pip install labml labml-app
```

### Hosting the experiments server

#### Prerequisites
Ensure that `MongoDB` is installed before starting the server. To install `MongoDB`, refer to the official
documentation [here](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/).

#### Starting the server
```sh
# Start the server on the default port (5005)
labml app-server

# To start the server on a different port, use the following command
labml app-server --port PORT
```

For a more comprehensive guide, please refer
to [this](https://github.com/labmlai/labml/blob/master/guides/server-setup.md).

You can access the UI by visiting `http://localhost:{port}` or, if set up on a different machine,
use `http://{server-ip}:{port}`.

### PyTorch example

```python
from labml import tracker, experiment

with experiment.record(name='sample', exp_conf=conf):
    for i in range(50):
        loss, accuracy = train()
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
```

### Distributed training example

```python
from labml import tracker, experiment

experiment.create(uuid=experiment.generate_uuid(),
                  name='distributed training sample',
                  distributed_rank=1,
                  distributed_world_size=8,
                  )
with experiment.start():
    for i in range(50):
        loss, accuracy = train()
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
```

### 📚 Documentation

* [Python API Reference](https://docs.labml.ai)
* [Samples](https://github.com/labmlai/labml/tree/master/samples)

##### Guides

* [API to create experiments](https://colab.research.google.com/github/labmlai/labml/blob/master/guides/experiment.ipynb)
* [Track training metrics](https://colab.research.google.com/github/labmlai/labml/blob/master/guides/tracker.ipynb)
* [Monitored training loop and other iterators](https://colab.research.google.com/github/labmlai/labml/blob/master/guides/monitor.ipynb)
* [API for custom visualizations](https://colab.research.google.com/github/labmlai/labml/blob/master/guides/analytics.ipynb)
* [Configurations management API](https://colab.research.google.com/github/labmlai/labml/blob/master/guides/configs.ipynb)
* [Logger for stylized logging](https://colab.research.google.com/github/labmlai/labml/blob/master/guides/logger.ipynb)

### 🖥 Screenshots

#### Formatted training loop output

<div align="center">
    <img src="https://raw.githubusercontent.com/vpj/lab/master/images/logger_sample.png" alt="Sample Logs"/>
</div>

#### Custom visualizations based on Tensorboard logs

<div align="center">
    <img src="https://raw.githubusercontent.com/vpj/lab/master/images/analytics.png" alt="Analytics"/>
</div>

## Tools

### [Training models on cloud](https://github.com/labmlai/labml/tree/master/remote)

```bash
# Install the package
pip install labml_remote

# Initialize the project
labml_remote init

# Add cloud server(s) to .remote/configs.yaml

# Prepare the remote server(s)
labml_remote prepare

# Start a PyTorch distributed training job
labml_remote helper-torch-launch --cmd 'train.py' --nproc-per-node 2 --env GLOO_SOCKET_IFNAME enp1s0
```

### [Monitoring hardware usage](https://github.com/labmlai/labml/blob/master/guides/hardware_monitoring.md)

```sh
# Install packages and dependencies
pip install labml psutil py3nvml

# Start monitoring
labml monitor
```

## Other Guides

#### [Setting up a local Ubuntu workstation for deep learning](https://github.com/labmlai/labml/blob/master/guides/local-ubuntu.md)

#### [Setting up a cloud computer for deep learning](https://github.com/labmlai/labml/blob/master/guides/remote-python.md)

## Citing

If you use LabML for academic research, please cite the library using the following BibTeX entry.

```bibtext
@misc{labml,
 author = {Varuna Jayasiri, Nipun Wijerathne},
 title = {labml.ai: A library to organize machine learning experiments},
 year = {2020},
 url = {https://labml.ai/},
}
```
