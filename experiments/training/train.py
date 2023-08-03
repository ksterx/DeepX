import inspect
from logging import getLogger

import hydra

from deepx.tasks import GAN, VAE, Classification, Segmentation, TrainingConfig
from deepx.tasks.core import DataModuleConfig, ModelConfig, TaskConfig
from deepx.utils.config import check_duplicate_keys

logger = getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    ds_cfg = cfg.dataset
    machine_cfg = cfg.machine
    task_cfg = cfg.task

    match task_cfg.name:
        case "classification":
            task_cls = Classification
        case "gan":
            task_cls = GAN
        case "segmentation":
            task_cls = Segmentation
        case "vae":
            task_cls = VAE
        case _:
            raise ValueError(f"Unknown task: {task_cfg.name}")

    # Parse config
    model_dict = {}
    task_dict = {}
    dm_dict = {"dm": ds_cfg.name}
    train_dict = {}

    cfgs = {
        "model": [task_cls.model_cfg, ModelConfig],
        "task": [task_cls.task_cfg, TaskConfig],
        "dm": [task_cls.dm_cfg, DataModuleConfig],
        "train": [TrainingConfig],
    }

    keys = {"model": [], "task": [], "dm": [], "train": []}
    for k, v in cfgs.items():
        for vv in v:
            sig = inspect.signature(vv)
            for kk in sig.parameters.keys():
                keys[k].append(kk)

    for c in [ds_cfg, machine_cfg, task_cfg]:
        for k, v in c.items():
            if k in keys["model"]:
                model_dict[k] = v
            elif k in keys["task"]:
                task_dict[k] = v
            elif k in keys["dm"]:
                dm_dict[k] = v
            elif k in keys["train"]:
                train_dict[k] = v
            elif k == "name":
                pass
            else:
                logger.warning(f"Unknown key: {k} in {c}")

    check_duplicate_keys(model_dict, task_dict, dm_dict, train_dict)

    model_cfg = cfgs["model"][0](**model_dict)
    task_cfg = cfgs["task"][0](**task_dict)
    dm_cfg = cfgs["dm"][0](**dm_dict)
    train_cfg = cfgs["train"][0](**train_dict)

    trainer = task_cls.trainer(
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        dm_cfg=dm_cfg,
        train_cfg=train_cfg,
    )

    trainer.train()


if __name__ == "__main__":
    main()
