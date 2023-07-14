from deepx.trainers import ImageGenTrainer

trainer = ImageGenTrainer(backbone="resnet18", model="gan", datamodule="mnist")
trainer.train(epochs=100, monitor="val_loss_g", logger="tensorboard")
