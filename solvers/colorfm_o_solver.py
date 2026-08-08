from __future__ import annotations

from collections.abc import Callable

import lightning as L
import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from torch.utils.data import DataLoader, TensorDataset

from models.velocity_mlp import MLP


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        return self.model(x, t, **extras)


class _FitProgressCallback(L.Callback):
    def __init__(
        self,
        total_steps: int,
        callback: Callable[[int, int], None] | None,
    ) -> None:
        super().__init__()
        self.total_steps = total_steps
        self.callback = callback
        self.update_interval = max(total_steps // 100, 1)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        completed = min(trainer.global_step, self.total_steps)
        if self.callback is not None and (
            completed % self.update_interval == 0
            or completed == self.total_steps
        ):
            self.callback(completed, self.total_steps)


class FlowSolver(L.LightningModule):
    def __init__(self, cfg, model: torch.nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model if model is not None else MLP(cfg.model)
        self.path = AffineProbPath(scheduler=CondOTScheduler())

    def get_train_tuple(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        return path_sample.x_t, path_sample.t, path_sample.dx_t

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        pairs = batch[0]
        x_0 = pairs[:, 0]
        x_1 = pairs[:, 1]

        x_t, t, target = self.get_train_tuple(x_0, x_1)
        prediction = self.model(x_t, t)

        distance = (x_1 - x_0).square().sum(dim=1).sqrt()
        loss = (target - prediction).square().sum(dim=1)
        # scaled v-loss
        loss = (loss / (1e-4 + distance)).mean()

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=False,
        )
        return loss

    def configure_optimizers(self):
        optimizer_name = str(self.cfg.train.optimizer)
        if optimizer_name != "Adam":
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        return torch.optim.Adam(
            self.parameters(),
            lr=float(self.cfg.train.lr),
            betas=tuple(self.cfg.train.betas),
        )

    @torch.inference_mode()
    def transform_pixels(
        self,
        pixels: torch.Tensor,
        chunk_size: int | None = None,
        sampling_steps: int | None = None,
        transfer_strength: float = 1.0,
    ) -> torch.Tensor:
        """Integrate the learned field with flow_matching.ODESolver."""
        if sampling_steps is not None and sampling_steps <= 0:
            raise ValueError("sampling_steps must be greater than zero.")
        if not 0.0 <= transfer_strength <= 1.0:
            raise ValueError("transfer_strength must be between zero and one.")
        if transfer_strength == 0.0:
            return pixels.detach().clamp(0, 1).cpu()

        self.eval()
        ode_solver = ODESolver(WrappedModel(self.model))
        method = str(self.cfg.inference.ode_method)
        if sampling_steps is None:
            step_size = float(self.cfg.inference.ode_step_size)
        else:
            step_size = transfer_strength / sampling_steps
        chunk_size = chunk_size or int(self.cfg.inference.chunk_size)
        time_grid = torch.tensor(
            [0.0, transfer_strength],
            device=self.device,
            dtype=pixels.dtype,
        )
        results = []

        for start in range(0, len(pixels), chunk_size):
            x_init = pixels[start : start + chunk_size].to(self.device)
            transformed = ode_solver.sample(
                x_init=x_init,
                time_grid=time_grid,
                method=method,
                step_size=step_size,
                return_intermediates=False,
            )
            results.append(transformed.clamp(0, 1).cpu())

        return torch.cat(results, dim=0)

    def transform_image(
        self,
        image: np.ndarray,
        sampling_steps: int | None = None,
        transfer_strength: float = 1.0,
    ) -> np.ndarray:
        """Apply the fitted field to an RGB uint8 image."""
        height, width = image.shape[:2]
        pixels = torch.from_numpy(image.astype(np.float32) / 255.0).reshape(-1, 3)
        output = self.transform_pixels(
            pixels,
            sampling_steps=sampling_steps,
            transfer_strength=transfer_strength,
        )
        output = output.reshape(height, width, 3).numpy()
        return np.rint(output * 255.0).astype(np.uint8)


def fit_flow_solver(
    pairs: torch.Tensor,
    cfg,
    progress_callback: Callable[[int, int], None] | None = None,
) -> FlowSolver:
    total_steps = int(cfg.train.total_steps)
    if total_steps <= 0:
        raise ValueError("cfg.train.total_steps must be greater than zero.")
    if len(pairs) == 0:
        raise ValueError("Color pairs cannot be empty.")

    L.seed_everything(int(cfg.seed), workers=True, verbose=False)
    torch.set_float32_matmul_precision("high")

    batch_size = min(int(cfg.data.batch_size), len(pairs))
    train_loader = DataLoader(
        TensorDataset(pairs),
        batch_size=batch_size,
        shuffle=bool(cfg.data.shuffle),
        num_workers=int(cfg.data.num_workers),
        drop_last=bool(cfg.data.drop_last),
    )

    solver = FlowSolver(cfg)
    progress = _FitProgressCallback(total_steps, progress_callback)
    accelerator = str(cfg.solver.get("accelerator", "auto"))
    devices = int(cfg.solver.get("devices", 1))
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=total_steps,
        max_epochs=int(cfg.train.max_epochs),
        logger=bool(cfg.solver.logger),
        enable_checkpointing=bool(cfg.solver.enable_checkpointing),
        enable_model_summary=bool(cfg.solver.enable_model_summary),
        enable_progress_bar=bool(cfg.solver.enable_progress_bar),
        num_sanity_val_steps=int(cfg.solver.num_sanity_val_steps),
        callbacks=[progress],
    )
    trainer.fit(model=solver, train_dataloaders=train_loader)
    return solver.eval()
