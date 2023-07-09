# Customize workflows

## Creating a new task, datamodule, or model

- `<new_model>.py` should be placed in `deepx/nn/`.

    ```python
    class NewModel(nn.Module):
        NAME = "new_model"
    ```

   Declare your model name as `NAME = <new_model>` at the top of the class, and also add it to `registered_models` in `deepx/nn/__init__.py`.

- `<new_task>.py` should be placed in `deepx/tasks/<task_category>/`.

    ```python
    from deepx.tasks import TaskX

    class NewTask(TaskX):
        NAME = "new_task"

    class NewTaskDM(DatamoduleX):

    ```

- `<new_datamodule>.py` (this is for specific dataset such as `MNIST`) should be placed in `deepx/tasks/<task_category>/datamodules/`.

    ```python
    from deepx.tasks import ClassificationDM

    class MNISTDM(ClassificationDM):
        NAME = "mnist"
    ```

- Register your task in `deepx/tasks/__init__.py`.
