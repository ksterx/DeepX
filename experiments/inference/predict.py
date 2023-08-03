import hydra
from imggen import ImageGenerationApp
from segmentation import SegmentationApp

from docs.examples.classification import ClassificationApp


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    initialdir = cfg.machine.log_dir
    match cfg.task.name:
        case "classification":
            app = ClassificationApp(initialdir)
        case "imggen":
            app = ImageGenerationApp(initialdir)
        case "segmentation":
            app = SegmentationApp(initialdir)
    app.launch()


if __name__ == "__main__":
    main()
