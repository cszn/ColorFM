#!/usr/bin/env python3
"""Generate directed ColorFM-O image pairs for training stage 1."""

import argparse
import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from datasets.color_pairs import FlowDataset
from solvers.colorfm_o_solver import fit_flow_solver


logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "colorfm_o.yaml"
SEGMENTATION_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--fit-steps", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps", "auto"], default="cuda")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--semantic-segmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def get_device(name, cuda_id):
    if name in {"cuda", "auto"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    if name != "auto":
        return torch.device("cuda:0" if name == "cuda" else name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in args.input_dir.expanduser().iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    content_paths = image_paths[args.start:args.end]

    config = OmegaConf.load(CONFIG_PATH)
    if args.fit_steps is not None:
        config.train.total_steps = args.fit_steps
    sampling_steps = args.sampling_steps or round(1.0 / config.inference.ode_step_size)

    device = get_device(args.device, args.cuda)
    config.solver.accelerator = "gpu" if device.type == "cuda" else device.type
    config.solver.devices = 1

    segmentation_model = None
    segmentation_processor = None
    if args.semantic_segmentation:
        segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
            SEGMENTATION_MODEL
        ).to(device).eval()
        segmentation_processor = SegformerImageProcessor.from_pretrained(
            SEGMENTATION_MODEL
        )

    total = len(content_paths) * (len(image_paths) - 1)
    current = 0
    generated = 0
    skipped = 0
    failed = 0
    progress = tqdm(
        total=total,
        desc="ColorFM stage 1",
        unit="pair",
        dynamic_ncols=True,
        disable=not args.progress,
    )

    for content_path in content_paths:
        for style_path in image_paths:
            if content_path == style_path:
                continue

            current += 1
            output_path = output_dir / f"{content_path.stem}_{style_path.stem}.png"
            if output_path.exists():
                skipped += 1
                progress.set_postfix_str(f"skip {output_path.name}", refresh=False)
                progress.update(1)
                continue

            progress.set_postfix_str(
                f"train {content_path.name} -> {style_path.name}", refresh=True
            )
            try:
                data_config = SimpleNamespace(
                    x_0=str(content_path),
                    x_1=str(style_path),
                    path=output_path.stem,
                    batch_size=4096,
                    num_workers=0,
                )
                dataset = FlowDataset(
                    data_config,
                    device=device,
                    SegModel=segmentation_model,
                    feature_extractor=segmentation_processor,
                    full=True,
                    seg_mode=args.semantic_segmentation,
                )
                pairs = dataset.pairs
                with open(os.devnull, "w") as quiet_output:
                    with redirect_stdout(quiet_output), redirect_stderr(quiet_output):
                        solver = fit_flow_solver(pairs, config).to(device).eval()

                with Image.open(content_path) as image:
                    content = np.asarray(
                        image.convert("RGB").resize((args.image_size, args.image_size))
                    )
                with torch.inference_mode():
                    result = solver.transform_image(
                        content,
                        sampling_steps=sampling_steps,
                        transfer_strength=1.0,
                    )
                Image.fromarray(result).save(output_path)
                generated += 1
            except Exception as error:
                failed += 1
                print(
                    f"[{current}/{total}] failed: {content_path.name} -> "
                    f"{style_path.name}: {error}",
                    file=sys.stderr,
                )
            finally:
                progress.set_postfix(
                    generated=generated,
                    skipped=skipped,
                    failed=failed,
                    refresh=False,
                )
                progress.update(1)

    progress.close()
    print(f"finished: generated={generated}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
