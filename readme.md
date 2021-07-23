<div align="center" style="margin-bottom: 100px;">

<!--
<img src="https://raw.githubusercontent.com/lab-ml/lab/master/images/lab_logo.png" width="150" alt="">
-->

<h1>Monitor deep learning model training and hardware usage from mobile.</h1>

[![PyPI - Python Version](https://badge.fury.io/py/labml.svg)](https://badge.fury.io/py/labml)
[![PyPI Status](https://pepy.tech/badge/labml)](https://pepy.tech/project/labml)
[![Join Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/)
[![Docs](https://img.shields.io/badge/labml-docs-blue)](https://docs.labml.ai/)
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai?ref_src=twsrc%5Etfw)

<img src="https://github.com/lab-ml/lab/blob/master/images/cover-dark.png" alt=""/>
</div>

### 🔥 Features

* Monitor running experiments from [mobile phone](https://github.com/lab-ml/app)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/9e7f39e047e811ebbaff2b26e3148b3d)
* Monitor [hardware usage on any computer](https://github.com/lab-ml/labml/blob/master/guides/hardware_monitoring.md) with a single command
* Integrate with just 2 lines of code (see examples below)
* Keeps track of experiments including infomation like git commit, configurations and hyper-parameters
* Keep Tensorboard logs organized
* [Dashboard](https://github.com/lab-ml/dashboard/) to locally browse and manage experiment runs
* Save and load checkpoints
* API for custom visualizations
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/samples/blob/master/labml_samples/pytorch/stocks/analysis.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vpj/poker/blob/master/kuhn_cfr/kuhn_cfr.ipynb)
* Pretty logs of training progress
* Open source! we also have a small hosted server for the mobile web app


### Installation

You can install this package using PIP.

```bash
pip install labml
```

### PyTorch example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ldu5tr0oYN_XcYQORgOkIY_Ohsi152fz?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/hnipun/monitoring-ml-model-training-on-your-mobile-phone)

```python
from labml import tracker, experiment

with experiment.record(name='sample', exp_conf=conf):
    for i in range(50):
        loss, accuracy = train()
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
```

### PyTorch Lightning example


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15aSPDwbKihDu_c3aFHNPGG5POjVlM2KO?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/hnipun/pytorch-lightning)

```python
from labml import experiment
from labml.utils.lightening import LabMLLighteningLogger

trainer = pl.Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=20, logger=LabMLLighteningLogger())

with experiment.record(name='sample', exp_conf=conf, disable_screen=True):
        trainer.fit(model, data_loader)

```


### TensorFlow 2.X Keras example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lx1dUG3MGaIDnq47HVFlzJ2lytjSa9Zy?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/hnipun/monitor-keras-model-training-on-your-mobile-phone)

```python
from labml import experiment
from labml.utils.keras import LabMLKerasCallback

with experiment.record(name='sample', exp_conf=conf):
    for i in range(50):
        model.fit(x_train, y_train, epochs=conf['epochs'], validation_data=(x_test, y_test),
                  callbacks=[LabMLKerasCallback()], verbose=None)
```

### [Monitoring hardware usage](https://github.com/lab-ml/labml/blob/master/guides/hardware_monitoring.md)

```sh
pip install labml psutil py3nvml
labml monitor
```

### 📚 Documentation

* [Python API Reference](https://docs.labml.ai)

##### Guides

* [API to create experiments](https://colab.research.google.com/github/lab-ml/labml/blob/master/guides/experiment.ipynb)
* [Track training metrics](https://colab.research.google.com/github/lab-ml/labml/blob/master/guides/tracker.ipynb)
* [Monitored training loop and other iterators](https://colab.research.google.com/github/lab-ml/labml/blob/master/guides/monitor.ipynb)
* [API for custom visualizations](https://colab.research.google.com/github/lab-ml/labml/blob/master/guides/analytics.ipynb)
* [Configurations management API](https://colab.research.google.com/github/lab-ml/labml/blob/master/guides/configs.ipynb)
* [Logger for stylized logging](https://colab.research.google.com/github/lab-ml/labml/blob/master/guides/logger.ipynb)

### 🖥 Screenshots

#### Formatted training loop output

<div align="center">
    <img src="https://raw.githubusercontent.com/vpj/lab/master/images/logger_sample.png" alt="Sample Logs"/>
</div>

#### Custom visualizations based on Tensorboard logs

<div align="center">
    <img src="https://raw.githubusercontent.com/vpj/lab/master/images/analytics.png" alt="Analytics"/>
</div>

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
