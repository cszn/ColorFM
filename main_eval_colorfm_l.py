#!/usr/bin/env python3
"""Evaluate a trained ColorFM-L model."""

import argparse
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf

from datasets.color_eval_pairs import get_eval_loader
from models.colorfm_l import ColorFM_L
from solvers.colorfm_l_solver import ColorFMLSolver


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "colorfm_l.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-path", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--full-resolution",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def load_checkpoint(solver, checkpoint):
    checkpoint = checkpoint.expanduser()
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")

    if "pytorch-lightning_version" in state:
        return str(checkpoint)

    solver.model.load_state_dict(state)
    print(f"Loaded model weights from {checkpoint}")
    return None


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.eval_path:
        cfg.data.eval_path = [str(path.expanduser()) for path in args.eval_path]
    if args.full_resolution is not None:
        cfg.data.eval_full_resolution = args.full_resolution

    checkpoint = args.checkpoint or (
        ROOT / "outputs" / str(cfg.exp_name) / "checkpoints" / "last.ckpt"
    )
    experiment_dir = ROOT / "outputs" / str(cfg.exp_name)
    eval_output_dir = (
        args.output_dir.expanduser()
        if args.output_dir
        else experiment_dir / "eval_images"
    )

    L.seed_everything(int(cfg.seed), workers=True)
    solver = ColorFMLSolver(cfg, ColorFM_L(cfg), eval_output_dir)
    eval_loader = get_eval_loader(cfg.data)
    lightning_checkpoint = load_checkpoint(solver, checkpoint)

    use_cuda = torch.cuda.is_available() and bool(cfg.cuda)
    trainer = L.Trainer(
        default_root_dir=experiment_dir,
        accelerator="gpu" if use_cuda else "cpu",
        devices=list(cfg.cuda) if use_cuda else 1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=bool(cfg.solver.enable_progress_bar),
    )
    trainer.validate(
        model=solver,
        dataloaders=eval_loader,
        ckpt_path=lightning_checkpoint,
    )


if __name__ == "__main__":
    main()
