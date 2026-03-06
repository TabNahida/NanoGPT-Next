from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms).to(dtype=x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, rope_base: float, rope_pct: float) -> None:
        super().__init__()
        rotary_dim = max(2, int(head_dim * rope_pct))
        rotary_dim -= rotary_dim % 2
        self.head_dim = head_dim
        self.rotary_dim = min(rotary_dim, head_dim)
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self.max_seq_len_cached:
            return
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]
        self.max_seq_len_cached = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(1)
        self._build_cache(seq_len)
        cos = self.cos_cached[:, :seq_len].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :seq_len].to(dtype=q.dtype, device=q.device)
        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    )


class GQAAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=config.use_bias)
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            rope_pct=config.rope_pct,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q, k = self.rope(q, k)
        if self.n_heads != self.n_kv_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=config.use_bias)
        self.w2 = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=config.use_bias)
        self.w3 = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.attn = GQAAttention(config)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MTPBlock(nn.Module):
    def __init__(self, d_model: int, use_bias: bool) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up = nn.Linear(d_model, d_model, bias=use_bias)
        self.down = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.up(h))
        return x + self.down(h)


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    main_loss: torch.Tensor | None
    mtp_losses: list[torch.Tensor]


class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.mtp_blocks = nn.ModuleList(
            [MTPBlock(config.d_model, use_bias=config.use_bias) for _ in range(config.mtp_heads)]
        )
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.apply(self._init_weights)

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return F.linear(hidden, self.tok_embeddings.weight)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = self.config.initializer_std
            if isinstance(module, nn.Linear) and module.out_features == self.config.d_model:
                std = std / math.sqrt(2 * self.config.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> ModelOutput:
        hidden = self.tok_embeddings(input_ids)
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self._project_logits(hidden)

        main_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None
        mtp_losses: list[torch.Tensor] = []

        if targets is not None:
            main_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            total_loss = main_loss
            mtp_hidden = hidden
            for idx, mtp_block in enumerate(self.mtp_blocks, start=1):
                mtp_hidden = mtp_block(mtp_hidden)
                if targets.size(1) <= idx:
                    break
                mtp_logits = self._project_logits(mtp_hidden[:, :-idx, :])
                mtp_targets = targets[:, idx:]
                mtp_loss = F.cross_entropy(
                    mtp_logits.reshape(-1, mtp_logits.size(-1)),
                    mtp_targets.reshape(-1),
                )
                mtp_losses.append(mtp_loss)
                weight = self.config.mtp_weight * (self.config.mtp_decay ** (idx - 1))
                total_loss = total_loss + (weight * mtp_loss)

        return ModelOutput(
            logits=logits,
            loss=total_loss,
            main_loss=main_loss,
            mtp_losses=mtp_losses,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len :]
            logits = self(idx_cond).logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        if was_training:
            self.train()
        return input_ids

    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
