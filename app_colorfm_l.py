from __future__ import annotations

import os
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from models.colorfm_l import ColorFM_L


REPO_DIR = Path(__file__).resolve().parent
# change ckpt path here
DEFAULT_WEIGHTS_PATH = REPO_DIR / "checkpoints" / "colorfm_l.pth"
WEIGHTS_PATH = Path(os.getenv("COLORFM_L_WEIGHTS", DEFAULT_WEIGHTS_PATH)).expanduser()

RESOLUTION_LIMITS = {
    "Original": 0,
    "4K": 3840 * 2160,
    "2K": 2560 * 1440,
    "1080p": 1920 * 1080,
    "720p": 1280 * 720,
}

_MODEL: ColorFM_L | None = None
_MODEL_DEVICE: torch.device | None = None
_MODEL_LOCK = threading.Lock()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("The ColorFM-L weight file is not a state dict.")
    return checkpoint


def load_model() -> tuple[ColorFM_L, torch.device]:
    global _MODEL, _MODEL_DEVICE

    if _MODEL is not None and _MODEL_DEVICE is not None:
        return _MODEL, _MODEL_DEVICE

    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_DEVICE is not None:
            return _MODEL, _MODEL_DEVICE
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                f"ColorFM-L weights were not found at {WEIGHTS_PATH}. "
                "Put colorfm_l.pth in checkpoints/ or set COLORFM_L_WEIGHTS."
            )

        device = get_device()
        model = ColorFM_L(cfg=None)
        model.load_state_dict(_load_state_dict(WEIGHTS_PATH), strict=True)
        model.to(device).eval()

        _MODEL = model
        _MODEL_DEVICE = device
        return model, device


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


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    array = np.ascontiguousarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def run_inference(
    content_image: np.ndarray | None,
    style_image: np.ndarray | None,
    transfer_strength: float,
    resolution: str,
) -> np.ndarray:
    if content_image is None or style_image is None:
        raise gr.Error("Please upload both a content image and a style image.")
    if not 0.0 <= float(transfer_strength) <= 1.0:
        raise gr.Error("Transfer strength must be between zero and one.")

    try:
        model, device = load_model()
        content = resize_to_limit(content_image, resolution)
        style = resize_to_limit(style_image, resolution)
        content_tensor = image_to_tensor(content, device)
        style_tensor = image_to_tensor(style, device)

        with _MODEL_LOCK, torch.inference_mode():
            transferred, _, _ = model(content_tensor, style_tensor)
            output = torch.lerp(
                content_tensor,
                transferred,
                float(transfer_strength),
            )
        return tensor_to_image(output)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


CUSTOM_CSS = """
#col-container {
    margin: 0 auto;
    max-width: 1100px;
}
"""


with gr.Blocks(css=CUSTOM_CSS, title="ColorFM-L") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 🎨 ColorFM-L")

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
            transfer_strength_input = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="🎚️ Transfer Strength",
                info="0 keeps the original colors; 1 uses the complete model output.",
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
                transfer_strength_input,
                resolution_input,
            ],
            outputs=output_image,
        )


if __name__ == "__main__":
    demo.launch()
