"""ONNX style-similarity metric for a reference/result image pair."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


METRICS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = METRICS_DIR / "StyleSimiliaryDiscriminator.onnx"


def _to_pil(image):
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            image = image[0]
        return to_pil_image(image.detach().cpu().clamp(0, 1)).convert("RGB")
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
        array = array * 255.0
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


@lru_cache(maxsize=None)
def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load and cache the style-similarity ONNX model."""

    return onnxruntime.InferenceSession(str(Path(model_path).expanduser()))


def compute_style_similarity(
    reference_image,
    generated_image,
    model_path=DEFAULT_MODEL_PATH,
):
    """Return the ONNX style-similarity score for two images."""

    size = (512, 512)
    reference = _to_pil(reference_image).resize(size, Image.Resampling.BILINEAR)
    generated = _to_pil(generated_image).resize(size, Image.Resampling.BILINEAR)
    reference_array = np.asarray(reference, dtype=np.float32)
    generated_array = np.asarray(generated, dtype=np.float32)

    session = load_model(model_path)
    score = session.run(
        ["score"],
        {"ref": reference_array, "img": generated_array},
    )[0]
    return float(np.asarray(score).squeeze())
