Installation & Setup
====================


Lab
----


**1. Clone and install**


.. code-block:: bash

    git clone git@github.com:vpj/lab.git
    cd lab
    pip install -e .

**2. To update run a git update**


.. code-block:: bash

    cd lab
    git pull

**3. Create a .lab.yaml file**

An empty file at the root of the project should be enough. You can set project level configs for 'check_repo_dirty' and 'path' 'check_repo_dirty' and 'path'.

Lab will store all experiment data in folder `logs/` relative to `.lab.yaml` file. If `path` is set in `.lab.yaml` then it will be stored in `[path]logs/` relative to `.lab.yaml` file.

You don't need the `.lab.yaml` file if you only plan on using the logger.


Lab Dashboard
--------------

**1. Clone and install**


.. code-block:: bash

  git clone git@github.com:vpj/lab_dashboard.git
  cd lab_dashboard
  git submodule init
  git submodule update
  ./install.sh


**2. To update run a git update**


.. code-block:: bash

  cd lab_dashboard
  git pull
  git submodule update
  ./install.sh


**3. Starting the server**

Navigate to the path of the project and run the following command to start the server.

.. code-block:: bash

  lab_dashboard


