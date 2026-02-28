from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nanogpt_next.eval.generate import generate
from nanogpt_next.models.decoder import DecoderOnlyLM
from nanogpt_next.tokenizer.wrapper import TokenizerWrapper
from nanogpt_next.utils.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg, _ = load_experiment_config(args.config, project_root=root)

    tokenizer = TokenizerWrapper.from_dir(root / cfg.data.tokenizer_dir)
    cfg.model.vocab_size = tokenizer.vocab_size

    model = DecoderOnlyLM(cfg.model)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = {
            key.removeprefix("model."): value
            for key, value in ckpt["state_dict"].items()
            if key.startswith("model.")
        }
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
