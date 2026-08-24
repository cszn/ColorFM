"""Evaluation image pairs for ColorFM-L."""

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ColorEvalDataset(Dataset):
    """Build content/style pairs from evaluation image folders."""

    def __init__(self, cfg):
        self.image_size = int(cfg.eval_image_size)
        self.full_resolution = bool(cfg.eval_full_resolution)
        self.pairs = []
        for eval_dir in cfg.eval_path:
            image_paths = sorted(
                path
                for path in Path(eval_dir).expanduser().iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for content_path in image_paths:
                for style_path in image_paths:
                    if content_path != style_path:
                        self.pairs.append((content_path, style_path))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        content_path, style_path = self.pairs[index]
        size = (self.image_size, self.image_size)
        content = Image.open(content_path).convert("RGB")
        if not self.full_resolution:
            content = content.resize(size)
        style = Image.open(style_path).convert("RGB").resize(size)
        return {
            "content": self.to_tensor(content),
            "style": self.to_tensor(style),
            "content_name": content_path.stem,
            "style_name": style_path.stem,
        }


def get_eval_loader(cfg):
    if not cfg.eval_path:
        return None
    num_workers = int(cfg.eval_num_workers)
    return DataLoader(
        ColorEvalDataset(cfg),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
