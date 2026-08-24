"""Approximate Lipschitz metric for a content/result image pair."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


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


def compute_lipschitz_metric(
    content_image,
    generated_image,
    num_samples=50000,
    seed=42,
):
    """Return the sampled maximum output/input color-distance ratio."""

    content = _to_pil(content_image)
    generated = _to_pil(generated_image)
    if content.size != generated.size:
        content = content.resize(generated.size, Image.Resampling.BILINEAR)

    content_pixels = np.asarray(content, dtype=np.float32).reshape(-1, 3) / 255.0
    generated_pixels = np.asarray(generated, dtype=np.float32).reshape(-1, 3) / 255.0

    np.random.seed(seed)
    indices = np.random.choice(
        len(content_pixels), (num_samples, 2), replace=True
    )
    input_distance = np.linalg.norm(
        content_pixels[indices[:, 0]] - content_pixels[indices[:, 1]], axis=1
    )
    output_distance = np.linalg.norm(
        generated_pixels[indices[:, 0]] - generated_pixels[indices[:, 1]], axis=1
    )
    valid = input_distance > 1e-6
    if not np.any(valid):
        return 0.0
    return float(np.max(output_distance[valid] / input_distance[valid]))
