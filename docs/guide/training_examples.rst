Training Examples
=================

Image Generation
----------------

MNIST
^^^^^

.. code-block:: bash

    cd experiments/training
    python imggenerator.py dataset=mnist machine=docker task=imggen

.. image:: ../_static/mnist.gif
    :align: center

LFW
^^^

.. code-block:: bash

    cd experiments/training
    python imggenerator.py dataset=lfw machine=docker task=imggen

.. image:: ../_static/lfw.gif
