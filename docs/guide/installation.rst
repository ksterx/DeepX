.. _installation:

Installation
============

Prerequisites
-------------

- Ubuntu 22.04 LTS / Windows WSL2 / MacOS
- Docker Engine (for Ubuntu)
- Docker Compose (for Ubuntu)

Ubuntu/WSL2 installation
------------------------

Build the docker image and run the container:

.. code-block:: bash

    cd envs
    docker compose up -d

After entering the container, install our python package:

.. code-block:: bash

    cd /workspace
    pip install -e .

MacOS installation
------------------

.. code-block:: bash

    cd envs
    pip install -r requirements.txt
    cd ..
    pip install -e .
