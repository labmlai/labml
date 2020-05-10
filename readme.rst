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

Lab is a library to improve your machine learning workflow and keep track of experiments.

.. about

We developed lab to speed-up our own machine learning workflow.
We kept it open source from the beginning.
Lab is small library (~4,000 lines of code), so anyone can dig into its codebase.
We make improvements to Lab based on our machine learning experience.
We also have made a bunch of improvements based feedback from other users.

.. who it is for

Lab started as a project to help individuals with their machine learning experiments.
It has come a long way since then, and from our experience,
it can help small research groups too.

.. state

We use lab in every internal project.
So, we have and will work on Lab actively.
When improve Lab, we might have to make breaking changes to it.
But, as the project is getting mature, breaking changes will be rare.

.. Organize Experiments

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

