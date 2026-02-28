from __future__ import annotations

import torch

from nanogpt_next.layers.attention import GQAAttention
from nanogpt_next.layers.rope import RotaryEmbedding


def test_rope_preserves_norm() -> None:
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32, partial_rotary_factor=1.0)
    q = torch.randn(2, 12, 4, 16)
    k = torch.randn(2, 12, 4, 16)
    q2, k2 = rope.apply_rotary(q, k)

    assert q2.shape == q.shape
    assert k2.shape == k.shape

    q_norm = torch.linalg.vector_norm(q[..., :16], dim=-1)
    q2_norm = torch.linalg.vector_norm(q2[..., :16], dim=-1)
    assert torch.allclose(q_norm, q2_norm, atol=1e-4, rtol=1e-4)


def test_gqa_forward_shape() -> None:
    attn = GQAAttention(
        hidden_size=64,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=64,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
    )
    x = torch.randn(2, 16, 64)
    y = attn(x)
    assert y.shape == x.shape
