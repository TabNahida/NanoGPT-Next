from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanogpt_next.layers.rope import RotaryEmbedding


class GQAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        rope_theta: float,
        partial_rotary_factor: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_repeat = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
        self.dropout = dropout

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, self.num_heads, self.head_dim)

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, self.num_kv_heads, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._shape_q(self.q_proj(x))
        k = self._shape_kv(self.k_proj(x))
        v = self._shape_kv(self.v_proj(x))

        q, k = self.rope.apply_rotary(q, k)

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=2)
            v = v.repeat_interleave(self.kv_repeat, dim=2)

        # SDPA expects [B, heads, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = attn_out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.hidden_size)
        return self.out_proj(out)
