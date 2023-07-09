import argparse

from deepx.trainers import ClassificationTrainer, LangModelTrainer, SegmentationTrainer

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, required=True)
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=100)
args = parser.parse_args()


match args.task:
    case "classification":
        trainer_cls = ClassificationTrainer
    case "segmentation":
        trainer_cls = SegmentationTrainer
    case "langmodel":
        trainer_cls = LangModelTrainer
    case _:
        raise ValueError(f"Task {args.task} not supported")

trainer = trainer_cls(args.model, args.dataset, batch_size=args.batch_size, num_workers=args.num_workers)
trainer.fit(epochs=args.epochs)
