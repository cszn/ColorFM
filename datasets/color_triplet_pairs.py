"""Training triplets for ColorFM-L stage 2."""

import random
from pathlib import Path

from lightning.pytorch.utilities.rank_zero import rank_zero_info
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ColorTripletDataset(Dataset):
    """Load content, style, and the corresponding ColorFM-O result."""

    def __init__(self, cfg):
        self.image_size = int(cfg.train_image_size)
        if len(cfg.image_dir) != len(cfg.styled_dir):
            raise ValueError("data.image_dir and data.styled_dir must have the same length")

        self.triplets = []
        for image_dir, styled_dir in zip(cfg.image_dir, cfg.styled_dir):
            image_paths = {
                path.stem: path
                for path in Path(image_dir).expanduser().iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }
            styled_paths = sorted(
                path
                for path in Path(styled_dir).expanduser().iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for styled_path in styled_paths:
                content_name, style_name = styled_path.stem.split("_", 1)
                self.triplets.append(
                    (image_paths[content_name], image_paths[style_name], styled_path)
                )

        self.to_tensor = T.ToTensor()
        self.color_aug = T.RandomApply(
            [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)],
            p=float(cfg.color_aug_probability),
        )

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        content_path, style_path, styled_path = self.triplets[index]
        content = Image.open(content_path).convert("RGB")
        style = Image.open(style_path).convert("RGB")
        styled = Image.open(styled_path).convert("RGB")

        size = (self.image_size, self.image_size)
        content = content.resize(size)
        style = style.resize(size)
        styled = styled.resize(size)

        if random.random() < 0.5:
            content = content.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            styled = styled.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            style = style.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        content = self.to_tensor(content)
        return {
            "content": self.color_aug(content),
            "style": self.to_tensor(style),
            "target": self.to_tensor(styled),
        }


def get_loader(cfg):
    num_workers = int(cfg.num_workers)
    dataset = ColorTripletDataset(cfg)
    rank_zero_info(f"Training dataset size: {len(dataset)} triplets")
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
