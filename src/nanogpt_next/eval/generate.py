from __future__ import annotations

import torch

from nanogpt_next.models.decoder import DecoderOnlyLM
from nanogpt_next.tokenizer.wrapper import TokenizerWrapper


@torch.no_grad()
def generate(
    model: DecoderOnlyLM,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int | None = 50,
    device: str | torch.device = "cpu",
) -> str:
    model.eval()
    model.to(device)

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.cfg.max_seq_len :]
        logits, _, _ = model(x_cond)
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < values[:, [-1]]] = -float("inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
