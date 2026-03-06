from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import OptimizerConfig


@dataclass(slots=True)
class OptimizerBundle:
    adamw: torch.optim.Optimizer
    muon: torch.optim.Optimizer | None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, object]:
        return {
            "adamw": self.adamw.state_dict(),
            "muon": None if self.muon is None else self.muon.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.adamw.load_state_dict(state_dict["adamw"])
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])

    def step(self, scaler: torch.cuda.amp.GradScaler | None = None) -> None:
        if scaler is not None and scaler.is_enabled():
            scaler.step(self.adamw)
            if self.muon is not None:
                scaler.step(self.muon)
            scaler.update()
            return
        self.adamw.step()
        if self.muon is not None:
            self.muon.step()

    def set_lrs(self, adamw_lr: float, muon_lr: float | None = None) -> None:
        for group in self.adamw.param_groups:
            group["lr"] = adamw_lr
        if self.muon is not None and muon_lr is not None:
            for group in self.muon.param_groups:
                group["lr"] = muon_lr


def build_optimizers(
    model: nn.Module,
    config: OptimizerConfig,
    device: torch.device,
) -> OptimizerBundle:
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []
    muon_params: list[nn.Parameter] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim == 2 and "tok_embeddings" not in name:
            muon_params.append(param)
            continue
        if param.ndim >= 2:
            adamw_decay.append(param)
        else:
            adamw_no_decay.append(param)

    adamw_groups = [
        {"params": adamw_decay, "weight_decay": config.adamw_weight_decay},
        {"params": adamw_no_decay, "weight_decay": 0.0},
    ]
    fused = bool(config.adamw_fused and device.type == "cuda")
    adamw = torch.optim.AdamW(
        adamw_groups,
        lr=config.adamw_lr,
        betas=config.adamw_betas,
        eps=config.adamw_eps,
        fused=fused,
    )

    muon: torch.optim.Optimizer | None = None
    if muon_params:
        muon = torch.optim.Muon(
            muon_params,
            lr=config.muon_lr,
            weight_decay=config.muon_weight_decay,
            momentum=config.muon_momentum,
            nesterov=config.muon_nesterov,
            ns_steps=config.muon_ns_steps,
            adjust_lr_fn=config.muon_adjust_lr_fn,
        )
    return OptimizerBundle(adamw=adamw, muon=muon)
