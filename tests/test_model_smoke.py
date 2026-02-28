from __future__ import annotations

import torch

from nanogpt_next.models.decoder import DecoderOnlyLM
from nanogpt_next.utils.config import ModelConfig


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        name="tiny",
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_multiplier=2.0,
        max_seq_len=32,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        dropout=0.0,
        rms_norm_eps=1e-5,
        tie_embeddings=True,
        mtp_num_future_tokens=2,
        mtp_lambda=0.2,
        use_activation_checkpointing=False,
    )


def test_forward_shapes() -> None:
    model = DecoderOnlyLM(_tiny_cfg())
    x = torch.randint(0, 128, (2, 32))
    logits, mtp_logits, hidden = model(x)
    assert logits.shape == (2, 32, 128)
    assert len(mtp_logits) == 2
    assert hidden.shape == (2, 32, 64)
