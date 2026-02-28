from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from nanogpt_next.layers.attention import GQAAttention
from nanogpt_next.layers.rmsnorm import RMSNorm
from nanogpt_next.layers.swiglu import SwiGLU
from nanogpt_next.utils.config import ModelConfig


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = GQAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            max_seq_len=cfg.max_seq_len,
            rope_theta=cfg.rope_theta,
            partial_rotary_factor=cfg.partial_rotary_factor,
            dropout=cfg.dropout,
        )
        self.ffn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.ffn = SwiGLU(
            hidden_size=cfg.hidden_size,
            ffn_multiplier=cfg.ffn_multiplier,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderOnlyLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        # MTP predicts tokens farther in the future; offset starts at +2.
        self.mtp_heads = nn.ModuleList(
            [
                nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
                for _ in range(cfg.mtp_num_future_tokens)
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.cfg.use_activation_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        if input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError(
                "Input sequence length "
                f"{input_ids.shape[1]} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        x = self.token_emb(input_ids)
        x = self._run_blocks(x)
        h = self.final_norm(x)
        logits = self.lm_head(h)
        mtp_logits = [head(h) for head in self.mtp_heads]
        return logits, mtp_logits, h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.cfg)
