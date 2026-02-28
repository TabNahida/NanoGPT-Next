from __future__ import annotations

import math

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl  # type: ignore

import torch
import torch.nn.functional as F

from nanogpt_next.models.decoder import DecoderOnlyLM
from nanogpt_next.optim.moun_adamw import MounAdamW
from nanogpt_next.utils.config import ModelConfig, TrainConfig


class NanoGPTLightningModule(pl.LightningModule):
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg", "train_cfg"])
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model = DecoderOnlyLM(model_cfg)

    def forward(self, input_ids: torch.Tensor):
        return self.model(input_ids)

    def _compute_losses(
        self,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch
        logits, mtp_logits, _ = self.model(input_ids)

        base_logits = logits[:, :-1, :].contiguous()
        base_targets = input_ids[:, 1:].contiguous()
        lm_loss = F.cross_entropy(base_logits.view(-1, base_logits.size(-1)), base_targets.view(-1))

        mtp_losses = []
        for idx, head_logits in enumerate(mtp_logits):
            offset = idx + 2
            if input_ids.size(1) <= offset:
                continue
            pred = head_logits[:, :-offset, :].contiguous()
            tgt = input_ids[:, offset:].contiguous()
            mtp_losses.append(F.cross_entropy(pred.view(-1, pred.size(-1)), tgt.view(-1)))

        mtp_loss = torch.stack(mtp_losses).mean() if mtp_losses else lm_loss.new_tensor(0.0)
        total = lm_loss + self.model_cfg.mtp_lambda * mtp_loss
        return total, lm_loss, mtp_loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx
        total, lm_loss, mtp_loss = self._compute_losses(batch)
        self.log("train/loss", total, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/lm_loss", lm_loss, on_step=True, on_epoch=False)
        self.log("train/mtp_loss", mtp_loss, on_step=True, on_epoch=False)
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False)
        return total

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx
        total, lm_loss, mtp_loss = self._compute_losses(batch)
        self.log("val/loss", total, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val/lm_loss", lm_loss, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val/mtp_loss", mtp_loss, on_step=False, on_epoch=True, sync_dist=False)
        return total

    def configure_optimizers(self):
        opt_cfg = self.train_cfg.optimizer
        if opt_cfg.name.lower() not in {"moun_adamw", "adamw"}:
            raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")

        optimizer = MounAdamW(
            params=self.parameters(),
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )

        sched_cfg = self.train_cfg.scheduler
        total_steps = max(1, int(self.train_cfg.max_steps))
        warmup_steps = int(total_steps * sched_cfg.warmup_ratio)
        min_lr_ratio = float(sched_cfg.min_lr_ratio)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(step / warmup_steps, 1e-8)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
