import hydra

from deepx.trainers import ClassificationTrainer


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    ds_cfg = cfg.dataset
    machine_cfg = cfg.machine
    task_cfg = cfg.task

    trainer = ClassificationTrainer(
        model=task_cfg.model,
        datamodule=ds_cfg.name,
        batch_size=task_cfg.batch_size,
        train_ratio=ds_cfg.train_ratio,
        num_workers=machine_cfg.num_workers,
        download=ds_cfg.download,
        lr=task_cfg.lr,
        beta1=task_cfg.beta1,
        beta2=task_cfg.beta2,
        loss_fn=task_cfg.loss_fn,
        optimizer=task_cfg.optimizer,
        scheduler=task_cfg.scheduler,
        root_dir=machine_cfg.root_dir,
        data_dir=machine_cfg.data_dir,
        log_dir=machine_cfg.log_dir,
        dropout=task_cfg.dropout,
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
    )


if __name__ == "__main__":
    main()
