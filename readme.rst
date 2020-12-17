.. image:: https://badge.fury.io/py/labml.svg
    :target: https://badge.fury.io/py/labml
.. image:: https://pepy.tech/badge/labml
    :target: https://pepy.tech/project/labml

LabML
=====

LabML lets you monitor AI model training on mobile phones.

.. image:: https://github.com/lab-ml/app/blob/master/images/labml-app.gif
   :width: 300px
   :alt: Mobile view 

You can install this package using PIP.

.. code-block:: console

    pip install labml


To push to mobile website, you need obtain a token from `web.lab-ml.com <https://web.lab-ml.com>`_
(Github `lab-ml/app <https://github.com/lab-ml/app/>`_), and save statistics with ``tracker.save``.

PyTorch example
^^^^^^^^^^^^^^^

.. code-block:: python

    from labml import tracker, experiment
  
    with experiment.record(name='sample', exp_conf=conf):
        for i in range(50):
            loss, accuracy = train()
            tracker.save(i, {'loss': loss, 'accuracy': accuracy})

Pytorch Lightening example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from labml import experiment
    from labml.utils.lightening import LabMLLighteningLogger

    trainer = pl.Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=20, logger=LabMLLighteningLogger())

    with experiment.record(name='sample', exp_conf=conf, disable_screen=True):
        trainer.fit(model, data_loader)

TensorFlow 2.X Keras example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from labml import experiment
    from labml.utils.keras import LabMLKerasCallback
  
    with experiment.record(name='sample', exp_conf=conf):
        for i in range(50):
            model.fit(x_train, y_train, epochs=conf['epochs'], validation_data=(x_test, y_test),
                      callbacks=[LabMLKerasCallback()], verbose=None)

You can read the guides about creating an  `experiment <http://lab-ml.com/guide/experiment.html>`_,
and saving statistics with `tracker <http://lab-ml.com/guide/tracker.html>`_ for details.

It automatically pushes data to Tensorboard, and you can keep your old experiments organized with the 
`LabML Dashboard <https://github.com/lab-ml/dashboard/>`_

.. image:: https://raw.githubusercontent.com/lab-ml/dashboard/master/images/screenshots/dashboard_table.png
   :width: 100%
   :alt: Dashboard Screenshot

All these software is 100% open source,
and your logs will be stored locally for Tensorboard and `LabML Dashboard <https://github.com/lab-ml/dashboard/>`_.
You will only be sending data away for `web.lab-ml.com <https://web.lab-ml.com>`_ if you include a token url.
This can also be `locally installed <https://github.com/lab-ml/app/>`_.

LabML can also keep track of git commits,
handle `configurations, hyper-parameters <http://lab-ml.com/guide/configs.html>`_,
save and load `checkpoints <http://lab-ml.com/guide/experiment.html>`_,
and providing pretty logs.

.. image:: https://raw.githubusercontent.com/vpj/lab/master/images/logger_sample.png
   :width: 50%
   :alt: Logger output

We also have an `API <https://lab-ml.com/guide/analytics.html>`_
to create `custom <https://github.com/lab-ml/samples/blob/master/labml_samples/pytorch/stocks/analysis.ipynb>`_
`visualizations <https://github.com/vpj/poker/blob/master/kuhn_cfr/kuhn_cfr.ipynb>`_
from artifacts and logs on Jupyter notebooks.

.. image:: https://raw.githubusercontent.com/vpj/lab/master/images/analytics.png
   :width: 50%
   :alt: Analytics

Links
-----

`💬 Slack workspace for discussions <https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/>`_

`📗 Documentation <http://lab-ml.com/>`_

`👨‍🏫 Samples <https://github.com/lab-ml/samples>`_


Citing LabML
------------

If you use LabML for academic research, please cite the library using the following BibTeX entry.

.. code-block:: bibtex

	@misc{labml,
	 author = {Varuna Jayasiri, Nipun Wijerathne},
	 title = {LabML: A library to organize machine learning experiments},
	 year = {2020},
	 url = {https://lab-ml.com/},
	}

