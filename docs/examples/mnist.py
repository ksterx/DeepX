from deepx.tasks.trainers import ClassificationTrainer

trainer = ClassificationTrainer(model="resnet18", datamodule="mnist")
trainer.train(epochs=1)
