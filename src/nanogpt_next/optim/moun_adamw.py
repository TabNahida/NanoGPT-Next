from __future__ import annotations

import torch


class MounAdamW(torch.optim.AdamW):
    """
    Practical AdamW-compatible optimizer placeholder for Moun+AdamW recipe.

    This preserves config compatibility and decoupled weight decay behavior.
    """

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
