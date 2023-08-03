# %%
from deepx.tasks.core import TrainingConfig
from deepx.tasks.segmentation import (
    SegmentationDMConfig,
    SegmentationModelConfig,
    SegmentationTaskConfig,
    SegmentationTrainer,
)

# %%
model_cfg = SegmentationModelConfig(
    model="unet",
    num_classes=21,
    in_channels=3,
)

# %%
task_cfg = SegmentationTaskConfig(
    lr=1e-4,
    loss_fn="ce",
    optimizer="adam",
    scheduler="cos",
    beta1=0.9,
    beta2=0.999,
)

# %%
dm_cfg = SegmentationDMConfig(
    dm="vocseg",
    batch_size=1,
    num_workers=2,
    train_ratio=0.8,
    data_dir="C:/Users/tomkj/Development/data",
    download=False,
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
    accelerator="cuda",
    devices=1,
    root_dir="C:/Users/tomkj/Development/DeepX",
    log_dir="C:/Users/tomkj/Development/mlruns",
)

# %%
trainer = SegmentationTrainer(
    model_cfg=model_cfg,
    task_cfg=task_cfg,
    dm_cfg=dm_cfg,
    train_cfg=train_cfg,
)

# %%
trainer.train()

# %%
