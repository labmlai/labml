.. image:: https://badge.fury.io/py/labml.svg
    :target: https://badge.fury.io/py/labml
.. image:: https://pepy.tech/badge/labml
    :target: https://pepy.tech/project/labml

LabML
=====

LabML is a library to track PyTorch experiments.

LabML keeps track of every detail of the experiments:
`source code <http://lab-ml.com/guide/experiment.html>`_,
`configurations, hyper-parameters <http://lab-ml.com/guide/configs.html>`_,
`checkpoints <http://lab-ml.com/guide/experiment.html>`_, 
`Tensorboard logs and other statistics <http://lab-ml.com/guide/tracker.html>`_.
LabML saves all these automatically in a clean folder structure.

This is an example usage of `Tracker <http://lab-ml.com/guide/tracker.html>`_

.. code-block:: python

    from labml import monit, tracker
    
    for epoch in monit.loop(50):
        for i in monit.iterate("Train", 10):
            time.sleep(1e-2)
            loss = 50 - epoch + np.random.randint(100) / 100
            tracker.save('loss.train', loss)
    
    if (epoch + 1) % 5 == 0:
        logger.log()
	
Here's the output,

.. image:: https://raw.githubusercontent.com/vpj/lab/master/images/logger_sample.png
   :width: 50%
   :alt: Logger output

Create an experiment and save the configurations with a couple of lines of codes,

.. code-block:: python

	from labml import experiment
	
	experiment.create(name='sin_wave')
	experiment.configs(configs)
	experiment.start()

View all your experiments locally with `Dashboard <https://github.com/vpj/lab_dashboard/>`_:

.. image:: https://raw.githubusercontent.com/lab-ml/dashboard/master/images/screenshots/dashboard_table.png
   :width: 100%
   :alt: Dashboard Screenshot

You can also `monitor your experiments on Slack <https://medium.com/@labml/labml-slack-integration-79519cf9c3a4>`_. 
When configured you will be receiving updates like following on a Slack thread.
Join our `Slack workspace <https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/>`_ to see samples.


.. image:: https://raw.githubusercontent.com/vpj/lab/master/images/slack_chart.png
   :width: 50%
   :alt: Example chart update on slack


Installation
------------

.. code-block:: console

    pip install labml

Links
-----

`💬 Slack workspace for discussions <https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/>`_

`📗 Documentation <http://lab-ml.com/>`_

`📑 Articles & Tutorials <https://medium.com/@labml/>`_

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

