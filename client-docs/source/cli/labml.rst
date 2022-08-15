labml
============

Synopsis
--------

**labml** [*options*] <*command*> [*command_specific_args*]

Options
-------

.. program:: labml

.. option:: -h, --help

   Display usage summary.

Commands
________

.. option:: app-server [*args*]

   Start `labml server <https://github.com/labmlai/labml/tree/master/app>`_.

   .. image:: https://github.com/labmlai/labml/raw/master/images/cover-dark.png
     :width: 400
     :alt: labml mobile UI

   .. important::

      Requires :program:`labml-app` to be installed.
      You can install it with :code:`pip install labml-app -U`.

   **args**:

   .. option:: --ip <ip_address>

      IP address to bind the local server instance.
      If unspecified, it will bind to 0.0.0.0,
      which makes the server accessible through all the assigned IPs of the computer.

      Default: 0.0.0.0

   .. option:: --port <port>

      Port to use for the local server instance.

      Default: 5005

   .. option:: -h, --help

      Display usage summary.

.. option:: capture [*args*]

   Capture the output of any command or program as an experiment.

   **args**: command to run. If no command is specified, data from the STDIN will be used; so you can pipe an output of a program to :code:`labml capture`.

   .. note::

      For example, :code:`labml capture python train.py`

.. option:: launch [*args*]

   Run a distributed training session with :program:`torch.distributed.launch`.

   **args**: command to run the distributed training session with :program:`torch.distributed.launch`.

.. option:: monitor

   `Monitor the hardware of the computer <https://github.com/labmlai/labml/blob/master/guides/hardware_monitoring.md>`_.

   .. image:: https://github.com/labmlai/labml/raw/master/guides/hardware.png
     :width: 150
     :alt: labml mobile UI

   .. important::

      | Requires :program:`psutil` to be installed.
      | Requires :program:`py3nvml` to be installed to monitor GPUs.

.. option:: service

   Setup the hardware monitoring as a service and start monitoring.

   .. note::

      Run :option:`monitor` first to make sure monitoring works without any issue, before setting up the service.

.. option:: service-run

   This is what gets called by the service installed usign :code:`labml service`.

   .. note::

      This is for internal use and you shouldn't run this manually.

.. option:: dashboard

   Open the dashboard to view experiments.

   .. deprecated:: 0.4.118
      Use :code:`labml server` instead.
