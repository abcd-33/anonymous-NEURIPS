"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os

# Use PyTorch backend for geomstats
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import os.path as osp
import sys
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import json
from glob import glob
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from manifm.datasets import get_loaders
from manifm.model_pl import ManifoldFMLitModule

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    logging.getLogger("pytorch_lightning").setLevel(logging.getLevelName("INFO"))

    if cfg.get("seed", None) is not None:
        pl.seed_everything(cfg.seed)

    # --- Dataset Loading ---
    train_loader, _, _ = get_loaders(cfg)

    # Construct model
    model = ManifoldFMLitModule(cfg)
    model.compile()
    print(model)

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="epoch-{epoch:03d}_step-{global_step}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            every_n_train_steps=cfg.get("ckpt_every", None),
        ),
        LearningRateMonitor(),
    ]

    slurm_plugin = SLURMEnvironment(auto_requeue=False)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cwd"] = os.getcwd()
    loggers = [CSVLogger(save_dir=".")]
    
    if cfg.use_wandb:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        loggers.append(
            WandbLogger(
                save_dir=".",
                name=f"{cfg.data.type}_{now}",
                project="ManiFM",
                log_model=False,
                config=cfg_dict,
                resume=True,
            )
        )

    # --- Trainer Configuration ---
    trainer = pl.Trainer(
        max_steps=cfg.optim.num_iterations,
        accelerator="gpu",
        devices=1,
        logger=loggers,
        # Removed val_check_interval and num_sanity_val_steps
        callbacks=callbacks,
        precision=cfg.get("precision", 32),
        gradient_clip_val=cfg.optim.grad_clip,
        plugins=slurm_plugin if slurm_plugin.detect() else None,
    )

    checkpoints = glob("checkpoints/**/*.ckpt", recursive=True)
    checkpoint = cfg.get("resume", None)
    if len(checkpoints) > 0:
        checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]

    # --- Training Loop ---
    trainer.fit(model, train_loader, ckpt_path=checkpoint)

    # Save final training metrics
    train_metrics = trainer.callback_metrics
    metric_dict = {k: float(v) for k, v in train_metrics.items()}

    with open("metrics.json", "w") as fout:
        print(json.dumps(metric_dict), file=fout)

    return metric_dict

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print(traceback.format_exc())
        sys.exit(1)