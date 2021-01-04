<div align="center" style="margin-bottom: 100px;">
<h1>LabML</h1>
<h2>Organize machine learning experiments and monitor training progress from mobile.</h2>

<img src="https://raw.githubusercontent.com/lab-ml/lab/master/images/lab_logo.png" width="200px" alt="">

[![PyPI - Python Version](https://badge.fury.io/py/labml.svg)](https://badge.fury.io/py/labml)
[![PyPI Status](https://pepy.tech/badge/labml)](https://pepy.tech/project/labml)
[![Join Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/)
[![Docs](https://img.shields.io/badge/labml-docs-blue)](https://lab-ml.com/)

<img src="https://github.com/lab-ml/app/blob/master/images/labml-app.gif" width="300" alt=""/>
</div>


You can install this package using PIP.

```bash
pip install labml
```

### PyTorch example

```python
from labml import tracker, experiment

with experiment.record(name='sample', exp_conf=conf):
    for i in range(50):
        loss, accuracy = train()
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
```

### PyTorch Lightning example

```python
from labml import experiment
from labml.utils.lightning import LabMLLightningLogger

trainer = pl.Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=20, logger=LabMLLightningLogger())

with experiment.record(name='sample', exp_conf=conf, disable_screen=True):
    trainer.fit(model, data_loader)
```

### TensorFlow 2.X Keras example

```python
from labml import experiment
from labml.utils.keras import LabMLKerasCallback

with experiment.record(name='sample', exp_conf=conf):
    for i in range(50):
        model.fit(x_train, y_train, epochs=conf['epochs'], validation_data=(x_test, y_test),
                  callbacks=[LabMLKerasCallback()], verbose=None)
```

### 🔥 Features

* Monitor running experiments from [mobile phone](https://github.com/lab-ml/app)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://web.lab-ml.com/run?uuid=9e7f39e047e811ebbaff2b26e3148b3d)
* Keeps track of experiments including infomation like git commit, configurations and hyper-parameters
* Keep Tensorboard logs organized
* [Dashboard](https://github.com/lab-ml/dashboard/)) to browse and manage experiment runs
* Save and load checkpoints
* Api to analyse the logs
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/samples/blob/master/labml_samples/pytorch/stocks/analysis.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vpj/poker/blob/master/kuhn_cfr/kuhn_cfr.ipynb)
* Pretty training loop logs
* Open source! we also have a small hosted server for the mobile web app

### 📚 Documentation

* [API to create experiments](https://lab-ml.com/guide/experiment.html)
* [Track training metrics](https://lab-ml.com/guide/tracker.html)
* [Monitored training loop and other iterators](https://lab-ml.com/guide/monit.html)
* [API for custom visualizations](https://lab-ml.com/guide/analytics.html)
* [Configurations management API](https://lab-ml.com/guide/configs.html)
* [Logger for stylized logging](https://lab-ml.com/guide/logger.html)

### 🖥 Screenshots

#### Dashboard

<div align="center">
    <img src="https://raw.githubusercontent.com/lab-ml/dashboard/master/images/screenshots/dashboard_table.png" alt="Dashboard Screenshot"/>
</div>

#### Formatted training loop output

<div align="center">
    <img src="https://raw.githubusercontent.com/vpj/lab/master/images/logger_sample.png" alt="Sample Logs"/>
</div>

#### Custom visualizations

<div align="center">
    <img src="https://raw.githubusercontent.com/vpj/lab/master/images/analytics.png" width="500" alt="Analytics"/>
</div>

## Links

[💬 Slack workspace for discussions](https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/)

[📗 Documentation](https://lab-ml.com/)

[👨‍🏫 Samples](https://github.com/lab-ml/samples)


## Citing LabML

If you use LabML for academic research, please cite the library using the following BibTeX entry.


```bibtext
@misc{labml,
 author = {Varuna Jayasiri, Nipun Wijerathne},
 title = {LabML: A library to organize machine learning experiments},
 year = {2020},
 url = {https://lab-ml.com/},
}
```
