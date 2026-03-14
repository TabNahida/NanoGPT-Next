from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, load_model_state_dict
from .config import DataConfig, ModelConfig
from .model import GPTModel
from .tokenizer import TokenizerSpec, build_tokenizer
from .utils import autocast_kwargs, detect_device, load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a NanoGPT-Next checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text. Falls back to config sample_prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16", "fp8"])
    parser.add_argument("--tokenizer-path", type=str, default="", help="Optional tokenizer override.")
    parser.add_argument("--tokenizer-backend", type=str, default="", help="Optional tokenizer backend override.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output text file.")
    return parser.parse_args()


def maybe_enable_float8(model: GPTModel, precision: str, device: torch.device) -> None:
    if precision != "fp8":
        return
    if device.type != "cuda":
        raise ValueError("FP8 checkpoint sampling requires a CUDA device.")
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    convert_to_float8_training(model, config=Float8LinearConfig())


def main() -> None:
    load_env_file()
    args = parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config_dict = checkpoint.get("config")
    if not isinstance(config_dict, dict):
        raise ValueError("Checkpoint does not contain embedded config metadata.")

    model_config = ModelConfig(**config_dict["model"])
    data_config = DataConfig(**config_dict.get("data", {}))
    trainer_config = config_dict.get("trainer", {})

    tokenizer_spec = TokenizerSpec(
        backend=args.tokenizer_backend or data_config.tokenizer_backend,
        path=args.tokenizer_path or data_config.tokenizer_path,
    )
    tokenizer = build_tokenizer(tokenizer_spec)

    device = detect_device(args.device)
    precision = trainer_config.get("precision", "bf16") if args.precision == "auto" else args.precision
    enable_autocast, autocast_dtype = autocast_kwargs(device, precision)
    if precision == "fp8":
        enable_autocast = device.type == "cuda"
        autocast_dtype = torch.bfloat16

    model = GPTModel(model_config)
    maybe_enable_float8(model, precision, device)
    load_model_state_dict(model, checkpoint["model"])
    model = model.to(device)
    model.eval()

    prompt = args.prompt or trainer_config.get("sample_prompt", "")
    if not prompt:
        raise ValueError("No prompt provided. Pass --prompt or set trainer.sample_prompt in the config.")

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    def autocast_context():
        if enable_autocast and autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    with torch.no_grad():
        with autocast_context():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
    text = tokenizer.decode(output_ids[0].tolist())
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

