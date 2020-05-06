.. image:: https://raw.githubusercontent.com/lab-ml/lab/master/images/lab_logo.png
   :width: 150
   :alt: Logo
   :align: center

Lab
===


`💬 Slack workspace for discussions <https://join.slack.com/t/labforml/shared_invite/zt-cg5iui5u-4cJPT7DUwRGqup9z8RHwhQ/>`_


Lab is a collection of small tools that work together to 
improve your machine learning workflow.

.. about

We developed lab to speed up our own machine learning workflow,
and have made it open source for others to use.
Lab is intentially kept simple and small so that not only us,
but anyone can dig into its codebase.
We add new features and make improvements to Lab based on our machine learning 
experience and feedback from other users.

.. who it is for

Lab started as a project to help individuals with their machine learning experiments.
It has come a long way since then, and from our experience, we believe it can help
small research groups too.

We work actively on this project, and will continue to do so,
since we are using it heavily for our own work.
So, Lab will rapidly get better.
The negative side of it is that we will be forced to make some breaking changes to it.
However, as the project is getting mature, we believe that breaking changes will be rare.

Organize Experiments
--------------------

Lab keeps track of every detail of the experiemnts:
source code,
configurations,
hyper-parements,
checkpoints, 
Tensorboard logs and other statistics.
All these are maintained automatically in a clean folder structure.

`Dashboard <https://github.com/vpj/lab_dashboard/>`_ is there, when you want to browse past experiemnts visually.

.. image:: https://raw.githubusercontent.com/vpj/lab/master/images/dashboard.png
   :width: 100%
   :alt: Dashboard Screenshot


Note
	`Dashboard <https://github.com/vpj/lab_dashboard/>`_ is a reletively new project and is improving very fast.
	As of now, you can view experiments, launch tensorboard, and delete unwanted experiments.

	We want to let users edit hyper-parameters and run new experiments directly from the dashboard,
	and do hyper-parameter searches.

	We plan on showing basic visualizations also on the dashboard.
	We are also playing around with using Jupyter Notebook based analytics.



Keep source code clean and encourage good coding practices
----------------------------------------------------------

Lab provides a bunch of utilities to help you keep your source code clean
by doing most of the overhead work for you.
This includes a range of utilites,
from monitored sections that let you split code into sections,
to a training loop that keeps and a tracker collect data for visualization.

.. The API of lab uses type hints and it works well with IDEs.


.. image:: https://raw.githubusercontent.com/vpj/lab/master/images/loop.gif
   :width: 100%
   :alt: Dashboard Screenshot

We introduced configurations to lab recently.
It lets you easily set hyper-parameters,
and encourage and assist researchers write reusable modular code.
It help keep away from bad practices like passing a large monolithic configuration object around,
and having a big class that does everything.
Using the Lab's configurations module, we were able to reuse a lot of code among our machine learning projects
and significantly improve the maintainability of the code base.

We have ve released some comomnly used configurable components such as ``TrainingLoop`` and ``Datasets``.
Any programmer can easily hack our components or write new reusable components to suite their requirements.

.. **Screenshot of a MNIST Sample**


`Slack workspace for discussions <https://join.slack.com/t/labforml/shared_invite/zt-cg5iui5u-4cJPT7DUwRGqup9z8RHwhQ/>`_
------------------------------------------------------------------------------------------------------------------------

If you have any feature suggestions, report any bugs or check feature updates, We have created a slack space for Lab. Please use this `URL <https://join.slack.com/t/labforml/shared_invite/zt-cg5iui5u-4cJPT7DUwRGqup9z8RHwhQ/>`_ to login.

Install
-------

.. code-block:: console

    pip install machine_learning_lab
