"""Lightning training module for ColorFM-L."""

from pathlib import Path

import lightning as L
import lpips
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from metrics.content_sim import compute_content_similarity
from metrics.lipschitz_metric import compute_lipschitz_metric
from metrics.style_sim import compute_style_similarity


class ColorFMLSolver(L.LightningModule):
    def __init__(self, cfg, model, eval_output_dir=None):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.eval_output_dir = eval_output_dir
        self.lpips_weight = float(cfg.train.lpips_weight)
        self.lpips_loss = lpips.LPIPS(net="vgg", verbose=False).requires_grad_(False)
        self.lpips_loss.eval()

    def _save_eval_images(self, batch, prediction, batch_idx):
        save_count = int(self.cfg.solver.eval_save_images)
        if save_count == 0 or (save_count != -1 and batch_idx >= save_count):
            return

        output_dir = self.eval_output_dir or (
            Path(self.trainer.default_root_dir) / "eval_images"
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for content_name, style_name, image in zip(
            batch["content_name"],
            batch["style_name"],
            prediction,
        ):
            filename = f"{content_name}_{style_name}.png"
            to_pil_image(image.detach().cpu().clamp(0, 1)).save(
                output_dir / filename
            )

    def forward(self, content, style):
        return self.model(content, style)

    def training_step(self, batch, batch_idx):
        prediction, _, _ = self.model(batch["content"], batch["style"])
        target = batch["target"]

        mse_loss = F.mse_loss(prediction, target)
        perceptual_loss = self.lpips_loss(
            prediction * 2.0 - 1.0,
            target * 2.0 - 1.0,
        ).mean()
        loss = mse_loss + self.lpips_weight * perceptual_loss

        self.log_dict(
            {
                "train/loss": loss,
                "train/mse": mse_loss,
                "train/lpips": perceptual_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=target.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        content = batch["content"]
        style = batch["style"]
        prediction, _, _ = self.model(content, style)
        self._save_eval_images(batch, prediction, batch_idx)

        metrics = {
            "eval/content_similarity": compute_content_similarity(
                content, prediction
            ),
            "eval/lipschitz": compute_lipschitz_metric(content, prediction),
            "eval/style_similarity": compute_style_similarity(style, prediction),
        }
        metrics = {
            name: torch.tensor(value, device=self.device)
            for name, value in metrics.items()
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=content.shape[0],
        )
        return metrics

    def configure_optimizers(self):
        train_cfg = self.cfg.train
        optimizer_name = str(train_cfg.optimizer).lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(train_cfg.lr),
                betas=tuple(train_cfg.betas),
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(train_cfg.lr),
                betas=tuple(train_cfg.betas),
                weight_decay=float(train_cfg.weight_decay),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {train_cfg.optimizer}")

        schedulers = []
        durations = []
        if train_cfg.warmup_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.001,
                    total_iters=int(train_cfg.warmup_epochs),
                )
            )
            durations.append(int(train_cfg.warmup_epochs))
        if train_cfg.constant_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=1.0,
                    total_iters=int(train_cfg.constant_epochs),
                )
            )
            durations.append(int(train_cfg.constant_epochs))
        if train_cfg.cosine_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(train_cfg.cosine_epochs),
                    eta_min=float(train_cfg.eta_min),
                )
            )
            durations.append(int(train_cfg.cosine_epochs))

        if not schedulers:
            return optimizer
        if len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            milestones = []
            elapsed = 0
            for duration in durations[:-1]:
                elapsed += duration
                milestones.append(elapsed)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=milestones,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
