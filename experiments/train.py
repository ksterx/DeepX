import hydra

from deepx.trainers import (
    ClassificationTrainer,
    ImageGenerationTrainer,
    SegmentationTrainer,
)


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg):
    ds_cfg = cfg.dataset
    machine_cfg = cfg.machine
    task_cfg = cfg.task

    match task_cfg.name:
        case "classification":
            trainer_cls = ClassificationTrainer
        case "imggen":
            trainer_cls = ImageGenerationTrainer
        case "segmentation":
            trainer_cls = SegmentationTrainer

    trainer = trainer_cls(
        model=task_cfg.model,
        datamodule=ds_cfg.name,
        batch_size=task_cfg.batch_size,
        num_workers=machine_cfg.num_workers,
        lr=task_cfg.lr,
        beta1=task_cfg.beta1,
        beta2=task_cfg.beta2,
        loss_fn=task_cfg.loss_fn,
        optimizer=task_cfg.optimizer,
        scheduler=task_cfg.scheduler,
        root_dir=machine_cfg.root_dir,
        data_dir=machine_cfg.data_dir,
        log_dir=machine_cfg.log_dir,
        dropout=task_cfg.dropout if hasattr(task_cfg, "dropout") else None,
        train_ratio=ds_cfg.train_ratio if hasattr(ds_cfg, "train_ratio") else None,
        download=ds_cfg.download if hasattr(ds_cfg, "download") else False,
        negative_slope=task_cfg.negative_slope
        if hasattr(task_cfg, "negative_slope")
        else None,
        latent_dim=task_cfg.latent_dim if hasattr(task_cfg, "latent_dim") else None,
        base_dim_g=task_cfg.base_dim_g if hasattr(task_cfg, "base_dim_g") else None,
        base_dim_d=task_cfg.base_dim_d if hasattr(task_cfg, "base_dim_d") else None,
    )
    trainer.train(
        ckpt_path=task_cfg.ckpt_path,
        epochs=task_cfg.epochs,
        stopping_patience=task_cfg.patience,
        max_depth=task_cfg.max_depth,
        benchmark=task_cfg.benchmark,
        debug=task_cfg.debug,
        monitor=task_cfg.monitor,
        monitor_max=task_cfg.monitor_max,
        logger=task_cfg.logger,
        accelerator=machine_cfg.accelerator,
        devices=machine_cfg.devices,
    )


if __name__ == "__main__":
    main()
