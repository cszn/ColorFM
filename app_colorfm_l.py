from __future__ import annotations

import gc
import hashlib
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

from models.colorfm_l import ColorFM_L


REPO_DIR = Path(__file__).resolve().parent
# change ckpt path here
DEFAULT_WEIGHTS_PATH = REPO_DIR / "checkpoints" / "colorfm_l.pth"
WEIGHTS_PATH = Path(os.getenv("COLORFM_L_WEIGHTS", DEFAULT_WEIGHTS_PATH)).expanduser()
DEFAULT_OUTPUT_DIR = Path(os.getenv("COLORFM_L_OUTPUT_DIR", "outputs/colorfm_l"))

DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu"]
DTYPE_CHOICES = ["auto", "bf16", "fp16", "fp32"]

RESOLUTION_LIMITS = {
    "Original": 0,
    "4K": 3840 * 2160,
    "2K": 2560 * 1440,
    "1080p": 1920 * 1080,
    "720p": 1280 * 720,
}

_MODEL: torch.nn.Module | None = None
_MODEL_DEVICE: torch.device | None = None
_MODEL_KEY: tuple[Any, ...] | None = None
_MODEL_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
_RESULT_CACHE_KEY: str | None = None
_RESULT_CACHE_CONTENT: np.ndarray | None = None
_RESULT_CACHE_TRANSFERRED: np.ndarray | None = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(choice: str) -> torch.device:
    choice = str(choice or "auto").lower()
    if choice == "auto":
        return get_device()
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise gr.Error("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise gr.Error("MPS was requested, but it is not available.")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    raise gr.Error(f"Unsupported device: {choice}")


def resolve_dtype(choice: str, device: torch.device) -> torch.dtype:
    choice = str(choice or "auto").lower()
    if choice == "auto":
        return torch.float32
    if device.type == "cpu":
        return torch.float32
    if choice == "fp32":
        return torch.float32
    if choice == "fp16":
        return torch.float16
    if choice == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise gr.Error("BF16 was requested, but the selected device does not support it.")
        return torch.bfloat16
    raise gr.Error(f"Unsupported dtype: {choice}")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_DIR / path
    return path.resolve()


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("The ColorFM-L weight file is not a state dict.")
    return checkpoint


def clear_model() -> None:
    global _MODEL, _MODEL_DEVICE, _MODEL_KEY
    global _RESULT_CACHE_KEY, _RESULT_CACHE_CONTENT, _RESULT_CACHE_TRANSFERRED

    _MODEL = None
    _MODEL_DEVICE = None
    _MODEL_KEY = None
    _RESULT_CACHE_KEY = None
    _RESULT_CACHE_CONTENT = None
    _RESULT_CACHE_TRANSFERRED = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(
    checkpoint: str | Path | None = None,
    device_choice: str = "auto",
    dtype_choice: str = "auto",
) -> tuple[torch.nn.Module, torch.device]:
    global _MODEL, _MODEL_DEVICE, _MODEL_KEY

    checkpoint_path = resolve_repo_path(checkpoint or WEIGHTS_PATH)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"ColorFM-L weights were not found at {checkpoint_path}. "
            "Choose a checkpoint in Settings or set COLORFM_L_WEIGHTS."
        )

    device = resolve_device(device_choice)
    dtype = resolve_dtype(dtype_choice, device)
    checkpoint_stat = checkpoint_path.stat()
    model_key = (
        str(checkpoint_path),
        checkpoint_stat.st_mtime_ns,
        checkpoint_stat.st_size,
        str(device),
        dtype,
    )

    if _MODEL is not None and _MODEL_DEVICE is not None and _MODEL_KEY == model_key:
        return _MODEL, _MODEL_DEVICE

    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_DEVICE is not None and _MODEL_KEY == model_key:
            return _MODEL, _MODEL_DEVICE

        clear_model()
        model = ColorFM_L(cfg=None)
        model.load_state_dict(_load_state_dict(checkpoint_path), strict=True)
        model.to(device=device, dtype=dtype).eval()

        _MODEL = model
        _MODEL_DEVICE = device
        _MODEL_KEY = model_key
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


def image_to_tensor(
    image: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    array = np.ascontiguousarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def tensor_to_float_image(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(
        tensor.detach().float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    )


def make_result_cache_key(
    content: np.ndarray,
    style: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    checkpoint_stat = checkpoint_path.stat()
    digest = hashlib.sha256()
    for image in (content, style):
        contiguous = np.ascontiguousarray(image)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    digest.update(str(checkpoint_path).encode())
    digest.update(str(checkpoint_stat.st_mtime_ns).encode())
    digest.update(str(checkpoint_stat.st_size).encode())
    digest.update(str(device).encode())
    digest.update(str(dtype).encode())
    return digest.hexdigest()


def blend_cached_result(
    content: np.ndarray,
    transferred: np.ndarray,
    transfer_strength: float,
) -> np.ndarray:
    content_float = content.astype(np.float32) / 255.0
    output = content_float + (transferred - content_float) * float(transfer_strength)
    return np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_result(output: np.ndarray, output_dir: str | Path, params: dict[str, Any]) -> Path:
    directory = resolve_repo_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    image_path = directory / f"{stamp}_colorfm_l.png"
    metadata_path = directory / f"{stamp}_colorfm_l.json"
    Image.fromarray(output).save(image_path)
    metadata_path.write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return image_path.resolve()


def _run_inference(
    content_image: np.ndarray | None,
    style_image: np.ndarray | None,
    transfer_strength: float,
    resolution: str,
    checkpoint: str | Path | None,
    output_dir: str | Path,
    device_choice: str,
    dtype_choice: str,
    save_output: bool,
) -> tuple[np.ndarray, str]:
    global _RESULT_CACHE_KEY, _RESULT_CACHE_CONTENT, _RESULT_CACHE_TRANSFERRED

    if content_image is None or style_image is None:
        raise gr.Error("Please upload both a content image and a style image.")
    if not 0.0 <= float(transfer_strength) <= 1.0:
        raise gr.Error("Transfer strength must be between zero and one.")
    if save_output and not str(output_dir).strip():
        raise gr.Error("Output Dir is required when Save Output is enabled.")

    checkpoint_path = resolve_repo_path(checkpoint or WEIGHTS_PATH)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"ColorFM-L weights were not found at {checkpoint_path}. "
            "Choose a checkpoint in Settings or set COLORFM_L_WEIGHTS."
        )

    started_at = time.perf_counter()
    device = resolve_device(device_choice)
    dtype = resolve_dtype(dtype_choice, device)
    content = resize_to_limit(content_image, resolution)
    style = resize_to_limit(style_image, resolution)
    cache_key = make_result_cache_key(
        content,
        style,
        checkpoint_path,
        device,
        dtype,
    )

    with _RUN_LOCK:
        cache_reused = (
            cache_key == _RESULT_CACHE_KEY
            and _RESULT_CACHE_CONTENT is not None
            and _RESULT_CACHE_TRANSFERRED is not None
        )
        if cache_reused:
            cached_content = _RESULT_CACHE_CONTENT
            cached_transferred = _RESULT_CACHE_TRANSFERRED
        else:
            model, device = load_model(
                checkpoint=checkpoint_path,
                device_choice=device_choice,
                dtype_choice=dtype_choice,
            )
            content_tensor = image_to_tensor(content, device, dtype)
            style_tensor = image_to_tensor(style, device, dtype)

            with _MODEL_LOCK, torch.inference_mode():
                transferred, _, _ = model(content_tensor, style_tensor)

            cached_content = np.ascontiguousarray(content).copy()
            cached_transferred = tensor_to_float_image(transferred)
            _RESULT_CACHE_KEY = cache_key
            _RESULT_CACHE_CONTENT = cached_content
            _RESULT_CACHE_TRANSFERRED = cached_transferred

        output_image = blend_cached_result(
            cached_content,
            cached_transferred,
            transfer_strength,
        )

    status_lines = [
        f"Finished in {time.perf_counter() - started_at:.2f}s",
        f"device: {device}",
        f"dtype: {str(dtype).replace('torch.', '')}",
        f"output size: {output_image.shape[1]}x{output_image.shape[0]}",
        f"full-strength result: {'cache reused' if cache_reused else 'model inference'}",
    ]
    if save_output:
        saved_path = save_result(
            output_image,
            output_dir,
            {
                "checkpoint": str(resolve_repo_path(checkpoint or WEIGHTS_PATH)),
                "resolution": resolution,
                "transfer_strength": float(transfer_strength),
                "device": str(device),
                "dtype": str(dtype),
                "cache_reused": cache_reused,
            },
        )
        status_lines.append(f"saved: {saved_path}")
    else:
        status_lines.append("saved: no")
    return output_image, "\n".join(status_lines)


def run_inference(
    content_image: np.ndarray | None,
    style_image: np.ndarray | None,
    transfer_strength: float,
    resolution: str,
    checkpoint: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device_choice: str = "auto",
    dtype_choice: str = "auto",
    save_output: bool = False,
) -> np.ndarray:
    try:
        output, _ = _run_inference(
            content_image,
            style_image,
            transfer_strength,
            resolution,
            checkpoint,
            output_dir,
            device_choice,
            dtype_choice,
            save_output,
        )
        return output
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def run_inference_with_status(
    content_image: np.ndarray | None,
    style_image: np.ndarray | None,
    transfer_strength: float,
    resolution: str,
    save_output: bool,
    checkpoint: str,
    output_dir: str,
    device_choice: str,
    dtype_choice: str,
) -> tuple[np.ndarray, str]:
    try:
        return _run_inference(
            content_image,
            style_image,
            transfer_strength,
            resolution,
            checkpoint,
            output_dir,
            device_choice,
            dtype_choice,
            save_output,
        )
    except gr.Error:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise gr.Error("CUDA out of memory. Try 1080p/720p, fp16, or CPU mode.") from exc
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
            with gr.Row():
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
                save_output_input = gr.Checkbox(
                    value=False,
                    label="💾 Save Output",
                    info="Save the PNG and run metadata only when enabled.",
                )

            with gr.Row():
                checkpoint_input = gr.Textbox(
                    value=str(Path(os.getenv("COLORFM_L_WEIGHTS", "checkpoints/colorfm_l.pth"))),
                    label="Checkpoint",
                )
                output_dir_input = gr.Textbox(
                    value=str(DEFAULT_OUTPUT_DIR),
                    label="Output Dir",
                    info="Used only when Save Output is enabled.",
                )

            with gr.Row():
                device_input = gr.Dropdown(
                    choices=DEVICE_CHOICES,
                    value=os.getenv("COLORFM_L_DEVICE", "auto"),
                    label="Device",
                )
                dtype_input = gr.Dropdown(
                    choices=DTYPE_CHOICES,
                    value=os.getenv("COLORFM_L_DTYPE", "auto"),
                    label="Dtype",
                    info="Auto preserves the original FP32 inference behavior.",
                )

        run_button = gr.Button("🚀 Start Color Transfer", variant="primary")
        output_image = gr.Image(
            label="✨ Result Image",
            type="numpy",
            format="png",
            height=450,
        )
        status_output = gr.Textbox(label="Status", lines=5, interactive=False)

        run_button.click(
            fn=run_inference_with_status,
            inputs=[
                content_input,
                style_input,
                transfer_strength_input,
                resolution_input,
                save_output_input,
                checkpoint_input,
                output_dir_input,
                device_input,
                dtype_input,
            ],
            outputs=[output_image, status_output],
        )


if __name__ == "__main__":
    demo.launch()
