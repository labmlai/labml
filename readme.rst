.. image:: https://raw.githubusercontent.com/lab-ml/lab/master/images/lab_logo.png
   :width: 150
   :alt: Logo
   :align: center

Lab
===


`💬 Slack workspace for discussions <https://join.slack.com/t/labforml/shared_invite/zt-cg5iui5u-4cJPT7DUwRGqup9z8RHwhQ/>`_

`📗 Documentation <http://lab-ml.com/>`_

`📑 Articles & Tutorials <https://medium.com/@labml/>`_

`👨‍🏫 Samples <https://github.com/lab-ml/samples>`_

Lab is a library to track PyTorch experiments.

Lab keeps track of every detail of the experiments:
`source code <http://lab-ml.com/guide/experiment.html>`_,
`configurations, hyper-parameters <http://lab-ml.com/guide/configs.html>`_,
`checkpoints <http://lab-ml.com/guide/experiment.html>`_, 
`Tensorboard logs and other statistics <http://lab-ml.com/guide/tracker.html>`_.
Lab saves all these automatically in a clean folder structure.

.. image: https://raw.githubusercontent.com/vpj/lab/master/images/loop.gif
   :width: 100%
   :alt: Logger output


You can use `Dashboard <https://github.com/vpj/lab_dashboard/>`_ to browse experiments.

.. image:: https://raw.githubusercontent.com/lab-ml/dashboard/master/images/screenshots/dashboard_table.png
   :width: 100%
   :alt: Dashboard Screenshot


.. 📝 Note
	`Dashboard <https://github.com/vpj/lab_dashboard/>`_ is a new project.
	With it, you can view experiments, launch TensorBoard, and delete unwanted experiments.

	We want to let users edit hyper-parameters, run new experiments,
	and do hyper-parameter searches from the dashboard.
	We plan on showing basic visualizations on the dashboard. 
	We are also playing around with using Jupyter Notebook for analytics.

Installation
------------

.. code-block:: console

    pip install machine_learning_lab

Citing Lab
----------

If you use Lab for academic research, please cite the library using the following BibTeX entry.

.. code-block:: bibtex

	@misc{lab,
	 author = {Varuna Jayasiri, Nipun Wijerathne},
	 title = {Lab: A library to organize machine learning experiments},
	 year = {2020},
	 url = {https://lab-ml.com/},
	}

