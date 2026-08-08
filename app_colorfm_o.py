from __future__ import annotations

import gc
import hashlib
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from datasets.color_pairs import FlowDataset
from solvers.colorfm_o_solver import FlowSolver, fit_flow_solver


REPO_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_DIR / "configs" / "colorfm_o.yaml"
DEFAULT_CONFIG = OmegaConf.load(CONFIG_PATH)
SEGMENTATION_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"

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
        dataset = FlowDataset(
            data_config,
            device=device,
            SegModel=segmentation_model,
            feature_extractor=segmentation_processor,
            full=True,
            seg_mode=use_segmentation,
        )

    del segmentation_model, segmentation_processor
    return dataset.pairs


def run_inference(
    content_image: np.ndarray | None,
    style_image: np.ndarray | None,
    fit_steps: int,
    sampling_steps: int,
    transfer_strength: float,
    use_segmentation: bool,
    resolution: str,
    progress=gr.Progress(),
) -> np.ndarray:
    global _CACHED_SOLVER, _CACHED_FIT_KEY

    if content_image is None or style_image is None:
        raise gr.Error("Please upload both a content image and a style image.")
    if int(fit_steps) <= 0:
        raise gr.Error("Fit steps must be greater than zero.")
    if int(sampling_steps) <= 0:
        raise gr.Error("Sampling steps must be greater than zero.")
    if not 0.0 <= float(transfer_strength) <= 1.0:
        raise gr.Error("Transfer strength must be between zero and one.")

    with _RUN_LOCK:
        try:
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

            if solver is None:
                progress(0, desc="Building color pairs")
                pairs = build_color_pairs(content, style, device, use_segmentation)

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
            else:
                progress(0, desc="Reusing cached flow model")

            progress(1.0, desc="Applying color transfer")
            return solver.transform_image(
                content,
                sampling_steps=int(sampling_steps),
                transfer_strength=float(transfer_strength),
            )
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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

        run_button = gr.Button("🚀 Start Color Transfer", variant="primary")
        output_image = gr.Image(
            label="✨ Result Image",
            type="numpy",
            format="png",
            height=450,
        )

        run_button.click(
            fn=run_inference,
            inputs=[
                content_input,
                style_input,
                fit_steps_input,
                sampling_steps_input,
                transfer_strength_input,
                segmentation_input,
                resolution_input,
            ],
            outputs=output_image,
        )


if __name__ == "__main__":
    demo.launch()
