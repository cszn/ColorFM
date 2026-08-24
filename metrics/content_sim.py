"""LDC edge-structure similarity for a content/result image pair."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision.transforms.functional import to_pil_image, to_tensor

from metrics.ldc import LDC, postprocess_edges


METRICS_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = METRICS_DIR / "ldc.pth"


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
def load_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device="cpu"):
    """Load and cache the LDC edge model."""

    model = LDC()
    path = Path(checkpoint_path).expanduser()
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _extract_edge(image, model, device):
    image = _to_pil(image).resize((512, 512), Image.Resampling.BILINEAR)
    image_tensor = to_tensor(image).unsqueeze(0).to(device) * 255.0
    mean = torch.tensor(
        [103.939, 116.779, 123.68], device=device
    ).view(1, 3, 1, 1)

    with torch.inference_mode():
        edges = model(image_tensor - mean)
    edge = postprocess_edges([value.float() for value in edges])
    return edge.astype(np.float32) / 255.0


def compute_content_similarity(
    content_image,
    generated_image,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    device=None,
):
    """Return SSIM between the LDC edges of two images."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(Path(checkpoint_path).expanduser()), str(device))
    content_edge = _extract_edge(content_image, model, device)
    generated_edge = _extract_edge(generated_image, model, device)
    return float(
        structural_similarity(content_edge, generated_edge, data_range=1.0)
    )
