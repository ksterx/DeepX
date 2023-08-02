# %%
from deepx.tasks.core import DataModuleConfig, TrainingConfig
from deepx.tasks.segmentation import (
    SegmentationConfig,
    SegmentationDMConfig,
    SegmentationModelConfig,
    SegmentationTrainer,
)

# %%
model_cfg = SegmentationModelConfig(model="resnet18", num_classes=10, in_channels=1)

# %%
task_cfg = SegmentationConfig(
    lr=1e-4, loss_fn="ce", optimizer="adam", scheduler="cos", beta1=0.9, beta2=0.999
)


# %%
dm_cfg = SegmentationDMConfig(
    dm="mnist",
    batch_size=32,
    num_workers=2,
    train_ratio=0.8,
    data_dir="/Users/ksterx/Development/PythonProjects/data",
)


# %%
train_cfg = TrainingConfig(
    ckpt_path=None,
    epochs=2,
    patience=5,
    max_depth=1,
    benchmark=True,
    debug=False,
    monitor_metric="val_loss",
    monitor_mode="min",
    logging=True,
    logger="mlflow",
    accelerator="cpu",
    devices=None,
    root_dir="/Users/ksterx/Development/PythonProjects/DeepX",
    log_dir="/Users/ksterx/Development/PythonProjects/mlruns",
)


# %%
trainer = SegmentationTrainer(
    model_cfg=model_cfg, task_cfg=task_cfg, dm_cfg=dm_cfg, train_cfg=train_cfg
)

# %%
trainer.train()


# %%
