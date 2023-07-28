.. _customizing:

Customizing workflow
===================

Creating a new model
--------------------

``<new_model>.py`` should be placed in ``deepx/nn/``.

.. code-block:: python

    class NewModel(nn.Module):
        NAME = "new_model"

Declare your model name as ``NAME = <new_model>`` at the top of the class, and also add this class to ``registered_models`` in ``deepx/nn/__init__.py``.

Creating a new task and datamodule
----------------------------------

- ``<new_task>.py`` should be placed in ``deepx/tasks``.

.. code-block:: python

    from .task import Task

    class NewTask(Task):
        NAME = "new_task"


- ``<new_datamodule>.py`` (this is for specific dataset such as ``MNIST``) should be placed in ``deepx/dms/data``.

.. code-block:: python

    from .dm import DataModule

    class NewTaskDM(Datamodule):


Creating a new dataset module
-----------------------------

``<new_dataset>.py`` should be placed in ``deepx/dms/data``. For example, ``MNIST`` is defined as follows:

.. code-block:: python

    from deepx.tasks import ClassificationDM

    class MNISTDM(ClassificationDM):
        NAME = "mnist"

Creating a new trainer
----------------------

``<new_trainer>.py`` should be placed in ``deepx/trainers/``.

.. code-block:: python

    from .trainer import Trainer

    class NewTrainer(Trainer):
        NAME = "new_trainer"

Register your algorithm
---------------------------------------------

Register your algorithm to ``deepx/__init__.py``.

.. code-block:: python

    from .dms import MNISTDM
    from .tasks import NewTask

    registered_tasks = {
        "new_task" {
            "name": NewTask,
            "datamodule": {
                "mnist": MNISTDM
            }
        }
    }
