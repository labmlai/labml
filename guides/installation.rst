Running Your Own Server Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Requirements: Python 3.7 and npm installed in your machine.

2. Clone the repository

.. code-block:: console

     git clone git@github.com:lab-ml/app.git

3. Install server and ui dependencies

.. code-block:: console

     make setup

4. Create ``app/server/app/setting.py`` similar to ``app/server/app/setting.example.py`` and ``app/ui/src/.env`` similar to ``app/ui/src/.env.example`` files and change the parameters accordingly.

5. For UI and server dev

.. code-block:: console

     make watch-ui
     make server-dev

6. For UI and server prod

.. code-block:: console

     make build-ui
     make server-prod
