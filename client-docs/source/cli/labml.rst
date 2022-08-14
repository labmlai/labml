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

   Start an instance of the labml server in the local machine.

   .. important::

      Requires :program:`labml-app` to be installed.

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

   Manually create an experiment using the provided data.

   **args**: command to run to capture the data. If no command is specified, data from the STDIN will be used.

.. option:: launch [*args*]

   Start a distributed training session and capture the data to a single experiment.

   **args**: command to run the distributed training session using :program:`torch.distributed.launch`.

.. option:: monitor

   Start monitoring the hardware of the computer.

   .. important::

      | Requires :program:`psutil` to be installed.
      | Requires :program:`py3nvml` to be installed to monitor GPUs.

.. option:: service

   Install the hardware monitoring as a service and start monitoring.

   .. note::

      Make sure to run :option:`monitor` first to make sure monitoring works without any issue.

.. option:: service-run

   Same as :option:`monitor`. But, this doesn't automatically open the browser to view the monitoring session
   after starting monitoring.

.. option:: dashboard

   Open the dashboard to view experiments.

   .. deprecated:: 0.4.118
      Please use labml.ai app instead.
