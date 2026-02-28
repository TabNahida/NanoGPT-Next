from __future__ import annotations

import torch

from nanogpt_next.layers.rmsnorm import RMSNorm
from nanogpt_next.layers.swiglu import SwiGLU


def test_rmsnorm_shape_grad() -> None:
    layer = RMSNorm(32)
    x = torch.randn(2, 4, 32, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_swiglu_shape_grad() -> None:
    layer = SwiGLU(hidden_size=64, ffn_multiplier=3.5)
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None
