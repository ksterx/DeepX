from deepx.trainers import ImageGenerationTrainer

trainer = ImageGenerationTrainer(
    model="dcgan", datamodule="cifar10", loss_fn="bce", lr=1e-4
)
trainer.train(epochs=100, monitor="val_loss_g")
