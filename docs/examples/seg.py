from deepx.tasks.trainers import SegmentationTrainer

trainer = SegmentationTrainer(model="unet", datamodule="vocseg")
trainer.train(debug=True)
