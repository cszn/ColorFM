#!/usr/bin/env python3
"""Train ColorFM-L from the image pairs generated in stage 1."""

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from datasets.color_eval_pairs import get_eval_loader
from datasets.color_triplet_pairs import get_loader
from models.colorfm_l import ColorFM_L
from solvers.colorfm_l_solver import ColorFMLSolver


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "colorfm_l.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    experiment_dir = args.output_dir.expanduser() / str(cfg.exp_name)
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(int(cfg.seed), workers=True)
    train_loader = get_loader(cfg.data)
    eval_loader = get_eval_loader(cfg.data)
    model = ColorFM_L(cfg)
    solver = ColorFMLSolver(cfg, model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="colorfm_l_{epoch:03d}",
        every_n_epochs=int(cfg.solver.checkpoint_every_n_epochs),
        save_top_k=-1,
        save_last=True,
    )
    logger = TensorBoardLogger(
        save_dir=experiment_dir,
        name="logs",
    )
    trainer = L.Trainer(
        default_root_dir=experiment_dir,
        accelerator="gpu",
        devices=list(cfg.cuda),
        max_epochs=int(cfg.train.max_epochs),
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=int(cfg.solver.log_every_n_steps),
        gradient_clip_val=float(cfg.solver.gradient_clip_val),
        enable_progress_bar=bool(cfg.solver.enable_progress_bar),
        num_sanity_val_steps=-1
    )
    resume_path = args.resume.expanduser() if args.resume else checkpoint_dir / "last.ckpt"
    trainer.fit(
        model=solver,
        train_dataloaders=train_loader,
        val_dataloaders=eval_loader,
        ckpt_path=str(resume_path) if resume_path.is_file() else None,
    )

    if trainer.is_global_zero:
        torch.save(solver.model.state_dict(), checkpoint_dir / "colorfm_l.pth")


if __name__ == "__main__":
    main()
