import argparse

from deepx.trainers import (
    ClassificationTrainer,
    ImageGenTrainer,
    LangModelTrainer,
    SegmentationTrainer,
)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, required=True)
parser.add_argument("-m", "--model", type=str, nargs="*", required=True)
parser.add_argument("-bb", "--backbone", type=str, default="resnet18")
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-r", "--root_dir", type=str, default="/workspace")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--download", action="store_true")
args = parser.parse_args()


match args.task:
    case "classification":
        trainer_cls = ClassificationTrainer
    case "segmentation":
        trainer_cls = SegmentationTrainer
    case "imagegen":
        trainer_cls = ImageGenTrainer
    case "langmodel":
        trainer_cls = LangModelTrainer
    case _:
        raise ValueError(f"Task {args.task} not supported")

if isinstance(args.model, str):
    args.model = [args.model]

for model in args.model:
    trainer = trainer_cls(
        model,
        args.dataset,
        backbone=args.backbone,
        batch_size=args.batch_size,
        download=args.download,
        root_dir=args.root_dir,
    )
    trainer.train(epochs=args.epochs, debug=args.debug)
