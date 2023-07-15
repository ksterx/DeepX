from deepx.trainers import ImageGenTrainer

trainer = ImageGenTrainer(backbone="mlp", model="gan", datamodule="cifar10", loss_fn="bce", lr=1e-4)
trainer.train(epochs=100, monitor="val_fid")
