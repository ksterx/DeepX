from deepx.tasks.trainers import ClassificationTrainer, LangModelTrainer

# trainer = ClassificationTrainer(model="resnet18", datamodule="mnist")
trainer = LangModelTrainer(model="lmtransformer", datamodule="penn")
trainer.train(debug=True)
