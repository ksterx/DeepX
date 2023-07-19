import hydra

from deepx.trainers import ClassificationTrainer


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    tcfg = cfg.task

    trainer = ClassificationTrainer(
        model=tcfg.model,
        datamodule=tcfg.datamodule,
        batch_size=tcfg.batch_size,
        train_ratio=tcfg.train_ratio,
        num_workers=tcfg.num_workers,
        download=tcfg.download,
        lr=tcfg.lr,
        loss_fn=tcfg.loss_fn,
        optimizer=tcfg.optimizer,
        scheduler=tcfg.scheduler,
        root_dir=cfg.root_dir,
        data_dir=cfg.data_dir,
        log_dir=cfg.log_dir,
        dropout=tcfg.dropout,
    )
    trainer.train(
        ckpt_path=tcfg.ckpt_path,
        epochs=tcfg.epochs,
        stopping_patience=tcfg.patience,
        max_depth=tcfg.max_depth,
        benchmark=tcfg.benchmark,
        debug=tcfg.debug,
        monitor=tcfg.monitor,
        monitor_max=tcfg.monitor_max,
        logger=tcfg.logger,
    )


if __name__ == "__main__":
    main()
