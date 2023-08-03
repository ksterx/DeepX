.. _inference:

Inference
=========

Image Classification
--------------------

.. code-block:: bash

    cd experiments/inference
    python predict.py task=classification dataset=cifar10

.. image:: ../_static/classifier.png
    :align: center

Image Segmentation
------------------

.. code-block:: bash

    cd experiments/inference
    python predict.py task=segmentation dataset=vocseg

.. image:: ../_static/segmenter.png


Image Generation
----------------

.. code-block:: bash

    cd experiments/inference
    python predict.py task=imggen dataset=lfw

.. image:: ../_static/imggenerator.png
