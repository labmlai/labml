<div align="center" style="margin-bottom: 100px;">
    
<h2>Mobile first web app to monitor PyTorch & TensorFlow model training</h2>
<h3>Relax while your models are training instead of sitting in front of a computer</h3>


[![PyPI - Python Version](https://badge.fury.io/py/labml-app.svg)](https://badge.fury.io/py/labml-app)
[![PyPI Status](https://pepy.tech/badge/labml-app)](https://pepy.tech/project/labml-app)
[![Docs](https://img.shields.io/badge/labml-docs-blue)](http://docs.labml.ai/)
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai?ref_src=twsrc%5Etfw)

<img src="https://github.com/labmlai/labml/blob/master/images/cover-dark.png" alt=""/>
</div>

This is an open-source library to push updates of your ML/DL model training to mobile. [Here's a sample experiment](https://app.labml.ai/run/39b03a1e454011ebbaff2b26e3148b3d)

You can host this on your own.
We also have a small [AWS instance running](https://app.labml.ai). and you are welcome to use it. Please consider using your own installation if you are running lots of experiments.

### Notable Features

* **Mobile first design:** web version, that gives you a great mobile experience on a mobile browser.
* **Model Gradients, Activations and Parameters:** Track and compare these indicators independently. We provide a separate analysis for each of the indicator types.
* **Summary and Detail Views:** Summary views would help you to quickly scan and understand your model progress. You can use detail views for more in-depth analysis.
* **Track only what you need:** You can pick and save the indicators that you want to track in the detail view. This would give you a customised summary view where you can focus on specific model indicators.
* **Standard ouptut:** Check the terminal output from your mobile. No need to SSH.

### [📚 How to track experiments?](https://github.com/labmlai/labml)

### How to run app locally?

Install the PIP package

```sh
pip install labml-app

```

Start the server

```sh
labml app-server
```

Set the web api url to `http://localhost:5000/api/v1/track?` when you run experiments.
You can also [set this on `.labml.yaml`](https://github.com/labmlai/labml/blob/master/guides/labml_yaml_file.md).

```python
from labml import tracker, experiment

with experiment.record(name='sample', token='http://localhost:5000/api/v1/track?'):
    for i in range(50):
        loss, accuracy = train()
        tracker.save(i, {'loss': loss, 'accuracy': accuracy})
```
