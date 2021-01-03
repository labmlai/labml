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

You can read the guides about creating an [🧪 experiment](https://lab-ml.com/guide/experiment.html),
and saving statistics with [📈 tracker](https://lab-ml.com/guide/tracker.html) for details.

LabML automatically pushes data to Tensorboard, and you can keep your old experiments organized locally with the 
[🎛 LabML Dashboard](https://github.com/lab-ml/dashboard/).

<img src="https://raw.githubusercontent.com/lab-ml/dashboard/master/images/screenshots/dashboard_table.png" alt="Dashboard Screenshot"/>

All these software is 100% open source . By default, the library will send experiment data to our hosted server
[web.lab-ml.com](https://web.lab-ml.com), you can also host and run [📱 LabML App server](https://github.com/lab-ml/app/>).


LabML also keeps track of git commits,
handle [configurations, hyper-parameters](https://lab-ml.com/guide/configs.html>),
save and load [checkpoints](https://lab-ml.com/guide/experiment.html),
and provide pretty logs.

<img src="https://raw.githubusercontent.com/vpj/lab/master/images/logger_sample.png" alt="Sample Logs"/>

We also have an [📊analytics API](https://lab-ml.com/guide/analytics.html)
to create custom visualizations from artifacts and logs on Jupyter notebooks.
Samples:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/samples/blob/master/labml_samples/pytorch/stocks/analysis.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vpj/poker/blob/master/kuhn_cfr/kuhn_cfr.ipynb)

<img src="https://raw.githubusercontent.com/vpj/lab/master/images/analytics.png" width="500px" alt="Analytics"/>


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
