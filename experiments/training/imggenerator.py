import hydra

from deepx.trainers import ImageGenerationTrainer


@hydra.main(config_path="../conf/task/image_generation/", config_name="config", version_base=None)
def main(cfg):
    ds_cfg = cfg.dataset

    trainer = ImageGenerationTrainer(
        backbone=ds_cfg.backbone,
        model=ds_cfg.model,
        datamodule=ds_cfg.datamodule,
        batch_size=ds_cfg.batch_size,
        train_ratio=ds_cfg.train_ratio,
        num_workers=ds_cfg.num_workers,
        download=ds_cfg.download,
        lr=ds_cfg.lr,
        loss_fn=ds_cfg.loss_fn,
        optimizer=ds_cfg.optimizer,
        scheduler=ds_cfg.scheduler,
        root_dir=cfg.root_dir,
        data_dir=cfg.data_dir,
        log_dir=cfg.log_dir,
        hidden_dim=ds_cfg.hidden_dim,
        negative_slope=ds_cfg.negative_slope,
        dropout=ds_cfg.dropout,
        latent_dim=ds_cfg.latent_dim,
        base_channels=ds_cfg.base_channels,
    )
    trainer.train(
        ckpt_path=ds_cfg.ckpt_path,
        epochs=ds_cfg.epochs,
        stopping_patience=ds_cfg.patience,
        max_depth=ds_cfg.max_depth,
        benchmark=ds_cfg.benchmark,
        debug=ds_cfg.debug,
        monitor=ds_cfg.monitor,
        monitor_max=ds_cfg.monitor_max,
        logger=ds_cfg.logger,
    )


if __name__ == "__main__":
    main()
