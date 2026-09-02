from __future__ import annotations

import gc
import hashlib
import json
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from datasets.color_pairs import FlowDataset
from solvers.colorfm_o_solver import FlowSolver, fit_flow_solver

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*",
    module=r"gradio\.routes",
)


CONFIG_PATH = REPO_DIR / "configs" / "colorfm_o.yaml"
DEFAULT_CONFIG = OmegaConf.load(CONFIG_PATH)
SEGMENTATION_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "colorfm_o"

RESOLUTION_LIMITS = {
    "Original": 0,
    "4K": 3840 * 2160,
    "2K": 2560 * 1440,
    "1080p": 1920 * 1080,
    "720p": 1280 * 720,
}

_RUN_LOCK = threading.Lock()
_CACHED_SOLVER: FlowSolver | None = None
_CACHED_FIT_KEY: str | None = None
_CACHED_SEGMENTATIONS: tuple[np.ndarray, np.ndarray] | None = None


class _CapturingFlowDataset(FlowDataset):
    """Expose the processed segmentation maps without changing FlowDataset."""

    content_segmentation: np.ndarray | None = None
    style_segmentation: np.ndarray | None = None

    def process_seg_map(self, image_seg, style_seg, seg_mode):
        content_segmentation, style_segmentation = super().process_seg_map(
            image_seg,
            style_seg,
            seg_mode,
        )
        if seg_mode:
            self.content_segmentation = (
                content_segmentation.detach().squeeze().cpu().numpy().astype(np.uint8)
            )
            self.style_segmentation = (
                style_segmentation.detach().squeeze().cpu().numpy().astype(np.uint8)
            )
        return content_segmentation, style_segmentation


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if np.issubdtype(image.dtype, np.floating) and max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_to_limit(image: np.ndarray, resolution: str) -> np.ndarray:
    image = normalize_image(image)
    max_pixels = RESOLUTION_LIMITS.get(resolution, 0)
    height, width = image.shape[:2]
    if max_pixels <= 0 or height * width <= max_pixels:
        return image

    scale = (max_pixels / (height * width)) ** 0.5
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_DIR / path
    return path.resolve()


def segmentation_palette() -> list[int]:
    palette: list[int] = []
    for label in range(256):
        value = label
        red = green = blue = 0
        for shift in range(8):
            red |= ((value >> 0) & 1) << (7 - shift)
            green |= ((value >> 1) & 1) << (7 - shift)
            blue |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
        palette.extend((red, green, blue))
    return palette


def save_segmentation_map(segmentation: np.ndarray, path: Path, size: tuple[int, int]) -> None:
    mask = Image.fromarray(np.asarray(segmentation, dtype=np.uint8), mode="P")
    mask.putpalette(segmentation_palette())
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    mask.save(path)


def save_result(
    output: np.ndarray,
    output_dir: str | Path,
    params: dict[str, Any],
    segmentation_maps: tuple[np.ndarray, np.ndarray] | None = None,
    content_size: tuple[int, int] | None = None,
    style_size: tuple[int, int] | None = None,
) -> dict[str, Path]:
    directory = resolve_repo_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    paths = {
        "result": directory / f"{stamp}_colorfm_o.png",
        "metadata": directory / f"{stamp}_colorfm_o.json",
    }
    Image.fromarray(output).save(paths["result"])

    if segmentation_maps is not None:
        if content_size is None or style_size is None:
            raise ValueError("Source image sizes are required when saving segmentation maps.")
        paths["content_segmentation"] = (
            directory / f"{stamp}_content_segmentation.png"
        )
        paths["style_segmentation"] = directory / f"{stamp}_style_segmentation.png"
        save_segmentation_map(
            segmentation_maps[0],
            paths["content_segmentation"],
            content_size,
        )
        save_segmentation_map(
            segmentation_maps[1],
            paths["style_segmentation"],
            style_size,
        )

    metadata = dict(params)
    metadata["files"] = {name: str(path.resolve()) for name, path in paths.items()}
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {name: path.resolve() for name, path in paths.items()}


def make_fit_cache_key(
    content: np.ndarray,
    style: np.ndarray,
    fit_steps: int,
    use_segmentation: bool,
) -> str:
    """Identify all inputs that affect the fitted velocity field."""
    digest = hashlib.sha256()
    for image in (content, style):
        contiguous = np.ascontiguousarray(image)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    digest.update(str(int(fit_steps)).encode())
    digest.update(str(bool(use_segmentation)).encode())
    digest.update(CONFIG_PATH.read_bytes())
    return digest.hexdigest()


def load_segmentation_model(device: torch.device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGMENTATION_MODEL_ID,
    ).to(device).eval()
    processor = SegformerImageProcessor.from_pretrained(
        SEGMENTATION_MODEL_ID,
    )
    return model, processor


def build_color_pairs(
    content: np.ndarray,
    style: np.ndarray,
    device: torch.device,
    use_segmentation: bool,
) -> torch.Tensor:
    pairs, _ = build_color_pairs_with_segmentations(
        content,
        style,
        device,
        use_segmentation,
    )
    return pairs


def build_color_pairs_with_segmentations(
    content: np.ndarray,
    style: np.ndarray,
    device: torch.device,
    use_segmentation: bool,
) -> tuple[torch.Tensor, tuple[np.ndarray, np.ndarray] | None]:
    segmentation_model = None
    segmentation_processor = None
    if use_segmentation:
        segmentation_model, segmentation_processor = load_segmentation_model(device)

    with tempfile.TemporaryDirectory(prefix="colorfm_o_") as temp_dir:
        temp_path = Path(temp_dir)
        content_path = temp_path / "content.png"
        style_path = temp_path / "style.png"
        Image.fromarray(content).save(content_path)
        Image.fromarray(style).save(style_path)

        data_config = SimpleNamespace(
            x_0=str(content_path),
            x_1=str(style_path),
            path="inference",
            batch_size=4096,
            num_workers=0,
        )
        dataset = _CapturingFlowDataset(
            data_config,
            device=device,
            SegModel=segmentation_model,
            feature_extractor=segmentation_processor,
            full=True,
            seg_mode=use_segmentation,
        )

    del segmentation_model, segmentation_processor
    segmentation_maps = None
    if use_segmentation:
        if dataset.content_segmentation is None or dataset.style_segmentation is None:
            raise RuntimeError("Semantic segmentation maps were not captured.")
        segmentation_maps = (
            dataset.content_segmentation,
            dataset.style_segmentation,
        )
    return dataset.pairs, segmentation_maps


def _run_inference(
    content_image: np.ndarray | None,
    style_image: np.ndarray | None,
    fit_steps: int,
    sampling_steps: int,
    transfer_strength: float,
    use_segmentation: bool,
    resolution: str,
    save_output: bool,
    save_segmentation: bool,
    output_dir: str | Path,
    progress=gr.Progress(),
) -> tuple[np.ndarray, str]:
    global _CACHED_SOLVER, _CACHED_FIT_KEY, _CACHED_SEGMENTATIONS

    if content_image is None or style_image is None:
        raise gr.Error("Please upload both a content image and a style image.")
    if int(fit_steps) <= 0:
        raise gr.Error("Fit steps must be greater than zero.")
    if int(sampling_steps) <= 0:
        raise gr.Error("Sampling steps must be greater than zero.")
    if not 0.0 <= float(transfer_strength) <= 1.0:
        raise gr.Error("Transfer strength must be between zero and one.")
    if save_output and not str(output_dir).strip():
        raise gr.Error("Output Dir is required when Save Output is enabled.")

    with _RUN_LOCK:
        try:
            started_at = time.perf_counter()
            device = get_device()
            content = resize_to_limit(content_image, resolution)
            style = resize_to_limit(style_image, resolution)

            fit_key = make_fit_cache_key(
                content,
                style,
                int(fit_steps),
                use_segmentation,
            )
            solver = _CACHED_SOLVER if fit_key == _CACHED_FIT_KEY else None
            fit_cache_reused = solver is not None

            if solver is None:
                progress(0, desc="Building color pairs")
                pairs, segmentation_maps = build_color_pairs_with_segmentations(
                    content,
                    style,
                    device,
                    use_segmentation,
                )

                config = OmegaConf.load(CONFIG_PATH)
                config.train.total_steps = int(fit_steps)
                solver = fit_flow_solver(
                    pairs=pairs,
                    cfg=config,
                    progress_callback=lambda step, total: progress(
                        (step, total),
                        desc=f"Optimizing ({step}/{total})",
                    ),
                )
                _CACHED_SOLVER = solver
                _CACHED_FIT_KEY = fit_key
                _CACHED_SEGMENTATIONS = segmentation_maps
            else:
                progress(0, desc="Reusing cached flow model")
                segmentation_maps = _CACHED_SEGMENTATIONS

            solver = solver.to(device).eval()

            progress(1.0, desc="Applying color transfer")
            output = solver.transform_image(
                content,
                sampling_steps=int(sampling_steps),
                transfer_strength=float(transfer_strength),
            )

            status_lines = [
                f"Finished in {time.perf_counter() - started_at:.2f}s",
                f"device: {device}",
                f"fit model: {'cache reused' if fit_cache_reused else 'optimized'}",
                f"semantic segmentation: {'on' if use_segmentation else 'off'}",
                f"output size: {output.shape[1]}x{output.shape[0]}",
            ]
            if save_output:
                saved_segmentations = (
                    segmentation_maps
                    if use_segmentation and save_segmentation
                    else None
                )
                paths = save_result(
                    output,
                    output_dir,
                    {
                        "config": str(CONFIG_PATH),
                        "fit_steps": int(fit_steps),
                        "sampling_steps": int(sampling_steps),
                        "transfer_strength": float(transfer_strength),
                        "semantic_segmentation": bool(use_segmentation),
                        "resolution": resolution,
                        "fit_cache_reused": fit_cache_reused,
                        "segmentation_maps_saved": saved_segmentations is not None,
                    },
                    segmentation_maps=saved_segmentations,
                    content_size=(content.shape[1], content.shape[0]),
                    style_size=(style.shape[1], style.shape[0]),
                )
                status_lines.append(f"saved: {paths['result']}")
                if saved_segmentations is not None:
                    status_lines.append(
                        "segmentation maps: "
                        f"{paths['content_segmentation']}, {paths['style_segmentation']}"
                    )
                elif save_segmentation and not use_segmentation:
                    status_lines.append(
                        "segmentation maps: not saved (Semantic Segmentation is off)"
                    )
            else:
                status_lines.append("saved: no")
                if save_segmentation:
                    status_lines.append(
                        "segmentation maps: not saved (Save Output is off)"
                    )
            return output, "\n".join(status_lines)
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_inference(
    content_image: np.ndarray | None,
    style_image: np.ndarray | None,
    fit_steps: int,
    sampling_steps: int,
    transfer_strength: float,
    use_segmentation: bool,
    resolution: str,
    progress=gr.Progress(),
    save_output: bool = False,
    save_segmentation: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> np.ndarray:
    output, _ = _run_inference(
        content_image,
        style_image,
        fit_steps,
        sampling_steps,
        transfer_strength,
        use_segmentation,
        resolution,
        save_output,
        save_segmentation,
        output_dir,
        progress,
    )
    return output


def run_inference_with_status(
    content_image: np.ndarray | None,
    style_image: np.ndarray | None,
    fit_steps: int,
    sampling_steps: int,
    transfer_strength: float,
    use_segmentation: bool,
    resolution: str,
    save_output: bool,
    save_segmentation: bool,
    output_dir: str,
    progress=gr.Progress(),
) -> tuple[np.ndarray, str]:
    return _run_inference(
        content_image,
        style_image,
        fit_steps,
        sampling_steps,
        transfer_strength,
        use_segmentation,
        resolution,
        save_output,
        save_segmentation,
        output_dir,
        progress,
    )


CUSTOM_CSS = """
#col-container {
    margin: 0 auto;
    max-width: 1100px;
}
"""


with gr.Blocks(css=CUSTOM_CSS, title="ColorFM-O") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 🎨 ColorFM-O")

        with gr.Row():
            content_input = gr.Image(
                label="🖼️ Content Image",
                type="numpy",
                image_mode="RGB",
                height=300,
            )
            style_input = gr.Image(
                label="🎨 Style Reference",
                type="numpy",
                image_mode="RGB",
                height=300,
            )

        with gr.Accordion(label="⚙️ Settings", open=False):
            with gr.Row():
                fit_steps_input = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=int(DEFAULT_CONFIG.train.total_steps),
                    step=1,
                    label="🔁 Fit Steps",
                )
                segmentation_input = gr.Checkbox(
                    value=False,
                    label="🧩 Semantic Segmentation",
                )
            with gr.Row():
                sampling_steps_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=round(1.0 / float(DEFAULT_CONFIG.inference.ode_step_size)),
                    step=1,
                    label="🪜 Sampling Steps",
                    info="More steps can improve ODE accuracy but increase inference time.",
                )
                transfer_strength_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    label="🎚️ Transfer Strength",
                    info="0 keeps the original colors; 1 follows the complete flow path.",
                )
            resolution_input = gr.Radio(
                choices=list(RESOLUTION_LIMITS),
                value="Original",
                label="📐 Maximum Output Resolution",
                info="Limit image size for faster inference. Original keeps the uploaded size.",
            )
            with gr.Row():
                save_output_input = gr.Checkbox(
                    value=False,
                    label="💾 Save Output",
                    info="Save the result PNG and run metadata when enabled.",
                )
                save_segmentation_input = gr.Checkbox(
                    value=False,
                    label="🧩 Save Segmentation Maps",
                    info=(
                        "Saved only when Save Output and Semantic Segmentation "
                        "are both enabled."
                    ),
                )
                output_dir_input = gr.Textbox(
                    value=str(DEFAULT_OUTPUT_DIR),
                    label="Output Dir",
                    info="Used only when Save Output is enabled.",
                )

        run_button = gr.Button("🚀 Start Color Transfer", variant="primary")
        output_image = gr.Image(
            label="✨ Result Image",
            type="numpy",
            format="png",
            height=450,
        )
        status_output = gr.Textbox(label="Status", lines=7, interactive=False)

        run_button.click(
            fn=run_inference_with_status,
            inputs=[
                content_input,
                style_input,
                fit_steps_input,
                sampling_steps_input,
                transfer_strength_input,
                segmentation_input,
                resolution_input,
                save_output_input,
                save_segmentation_input,
                output_dir_input,
            ],
            outputs=[output_image, status_output],
        )


if __name__ == "__main__":
    demo.launch()
