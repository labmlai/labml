<div align="center" style="margin-bottom: 100px;">

<h1>Monitor deep learning model training and hardware usage from mobile.</h1>

[![PyPI - Python Version](https://badge.fury.io/py/labml.svg)](https://badge.fury.io/py/labml)
[![PyPI Status](https://pepy.tech/badge/labml)](https://pepy.tech/project/labml)
[![Docs](https://img.shields.io/badge/labml-docs-blue)](https://docs.labml.ai/)
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai?ref_src=twsrc%5Etfw)

<img src="https://github.com/labmlai/labml/blob/master/images/cover-dark.png" alt=""/>
</div>

### 🔥 Features

* Monitor running experiments from mobile phone or laptop
* Monitor hardware usage on any computer
  with a single command
* Integrate with just 2 lines of code (see examples below)
* Keeps track of experiments including infomation like git commit, configurations and hyper-parameters
* API for custom visualizations
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/labml/blob/master/samples/stocks/analysis.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vpj/poker/blob/master/kuhn_cfr/kuhn_cfr.ipynb)
* Pretty logs of training progress
* Open source!

### Hosting the experiments server

#### Prerequisites

To install `MongoDB`, refer to the official
   documentation [here](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/).

#### Installation

Install the package using pip:

```bash
pip install labml-app
```

#### Starting the server

```sh
# Start the server on the default port (5005)
labml app-server

# To start the server on a different port, use the following command
labml app-server --port PORT
```

***Optional: to setup and configure Nginx in your server, please refer
to [this](https://github.com/labmlai/labml/blob/master/guides/server-setup.md).***

You can access the user interface either by visiting `http://localhost:{port}` or, if configured on a separate machine,
by navigating to `http://{server-ip}:{port}`.

### Monitor Experiments

#### Installation

1. Install the package using pip.

```bash
pip install labml
```

2. Create a file named `.labml.yaml` at the top level of your project folder, and add the following line to the file:

```yaml
app_url: http://localhost:{port}/api/v1/default

# If you are setting up the project on a different machine, include the following line instead,
app_url: http://{server-ip}:{port}/api/v1/default
```

#### PyTorch example

```python
from labml import tracker, experiment

with experiment.record(name='sample', exp_conf=conf):
    for i in range(50):
        loss, accuracy = train()
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
```

#### Distributed training example

```python
from labml import tracker, experiment

uuid = experiment.generate_uuid() # make sure to sync this in every machine
experiment.create(uuid=uuid,
                  name='distributed training sample',
                  distributed_rank=0,
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

[//]: # (## Tools)

[//]: # ()
[//]: # (### [Training models on cloud]&#40;https://github.com/labmlai/labml/tree/master/remote&#41;)

[//]: # ()
[//]: # (```bash)

[//]: # (# Install the package)

[//]: # (pip install labml_remote)

[//]: # ()
[//]: # (# Initialize the project)

[//]: # (labml_remote init)

[//]: # ()
[//]: # (# Add cloud server&#40;s&#41; to .remote/configs.yaml)

[//]: # ()
[//]: # (# Prepare the remote server&#40;s&#41;)

[//]: # (labml_remote prepare)

[//]: # ()
[//]: # (# Start a PyTorch distributed training job)

[//]: # (labml_remote helper-torch-launch --cmd 'train.py' --nproc-per-node 2 --env GLOO_SOCKET_IFNAME enp1s0)

[//]: # (```)

### [Monitoring hardware usage](https://github.com/labmlai/labml/blob/master/guides/hardware_monitoring.md)

```sh
# Install packages and dependencies
pip install labml psutil py3nvml

# Start monitoring
labml monitor
```

[//]: # (## Other Guides)

[//]: # ()
[//]: # (#### [Setting up a local Ubuntu workstation for deep learning]&#40;https://github.com/labmlai/labml/blob/master/guides/local-ubuntu.md&#41;)

[//]: # ()
[//]: # (#### [Setting up a cloud computer for deep learning]&#40;https://github.com/labmlai/labml/blob/master/guides/remote-python.md&#41;)

## Citing

If you use LabML for academic research, please cite the library using the following BibTeX entry.

```bibtext
@misc{labml,
 author = {Varuna Jayasiri, Nipun Wijerathne, Adithya Narasinghe, Lakshith Nishshanke},
 title = {labml.ai: A library to organize machine learning experiments},
 year = {2020},
 url = {https://labml.ai/},
}
```
