from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_multiplier: float, dropout: float = 0.0) -> None:
        super().__init__()
        inner = int(hidden_size * ffn_multiplier)
        # Keep multiples of 256 for accelerator-friendly kernels.
        inner = ((inner + 255) // 256) * 256
        self.gate = nn.Linear(hidden_size, inner, bias=False)
        self.up = nn.Linear(hidden_size, inner, bias=False)
        self.down = nn.Linear(inner, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.up(x)
        x = self.down(x)
        return self.dropout(x)
