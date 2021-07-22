<div align="center" style="margin-bottom: 100px;">
    
<h2>Mobile first web app to monitor PyTorch & TensorFlow model training</h2>
<h3>Relax while your models are training instead of sitting in front of a computer</h3>


[![PyPI - Python Version](https://badge.fury.io/py/labml.svg)](https://badge.fury.io/py/labml)
[![PyPI Status](https://pepy.tech/badge/labml)](https://pepy.tech/project/labml)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/)
[![Docs](https://img.shields.io/badge/labml-docs-blue)](http://lab-ml.com/)
[![Twitter](https://img.shields.io/twitter/url.svg?label=Follow%20%40LabML&style=social&url=https%3A%2F%2Ftwitter.com%2FLabML)](https://twitter.com/labmlai?ref_src=twsrc%5Etfw)

<img src="https://github.com/labmlai/labml/blob/master/images/cover-dark.png" alt=""/>
</div>

This is an open-source library to push updates of your ML/DL model training to mobile. [Here's a sample experiment](https://app.labml.ai/run/39b03a1e454011ebbaff2b26e3148b3d)

[You can host this on your own](https://github.com/lab-ml/app/blob/master/docs/installation.rst).
We also have a small [AWS instance running](https://app.labml.ai). and you are welcome to use it. Please consider using your own installation if you are running lots of
experiments. Thanks.

### Notable Features

* **Mobile first design:** web version, that gives you a great mobile experience on a mobile browser.
* **Model Gradients, Activations and Parameters:** Track and compare these indicators independently. We provide a separate analysis for each of the indicator types.
* **Summary and Detail Views:** Summary views would help you to quickly scan and understand your model progress. You can use detail views for more in-depth analysis.
* **Track only what you need:** You can pick and save the indicators that you want to track in the detail view. This would give you a customised summary view where you can focus on specific model indicators.
* **Standard ouptut:** Check the terminal output from your mobile. No need to SSH.

### How to use it ?
1. Install the [labml client library](https://github.com/lab-ml/labml).

```
pip install labml
```

2. Start pushing updates to the app  [with two lines of code](http://lab-ml.com/guide/tracker.html). Refer to the examples below.
3. Click on the link printed in the terminal to open the app. [![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/9e7f39e047e811ebbaff2b26e3148b3d)


### How to run app locally

```sh
pip install labml-app

```

```python
from labml import tracker, experiment

with experiment.record(name='sample', token='http://localhost:5000/api/v1/track?'):
    for i in range(50):
        loss, accuracy = train()
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
```


### Examples

1. Pytorch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ldu5tr0oYN_XcYQORgOkIY_Ohsi152fz?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/hnipun/monitoring-ml-model-training-on-your-mobile-phone)

```python
from labml import tracker, experiment

with experiment.record(name='sample', exp_conf=conf):
    for i in range(50):
        loss, accuracy = train()
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
```

2. PyTorch Lightning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15aSPDwbKihDu_c3aFHNPGG5POjVlM2KO?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/hnipun/pytorch-lightning)

```python
from labml import experiment
from labml.utils.lightening import LabMLLighteningLogger

trainer = pl.Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=20, logger=LabMLLighteningLogger())

with experiment.record(name='sample', exp_conf=conf, disable_screen=True):
        trainer.fit(model, data_loader)

```

3. TensorFlow 2.0 Keras [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lx1dUG3MGaIDnq47HVFlzJ2lytjSa9Zy?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/hnipun/monitor-keras-model-training-on-your-mobile-phone)

```python
from labml import experiment
from labml.utils.keras import LabMLKerasCallback

with experiment.record(name='sample', exp_conf=conf):
    for i in range(50):
        model.fit(x_train, y_train, epochs=conf['epochs'], validation_data=(x_test, y_test),
                  callbacks=[LabMLKerasCallback()], verbose=None)
```
