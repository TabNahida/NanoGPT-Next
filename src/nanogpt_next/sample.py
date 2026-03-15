from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint, load_model_state_dict
from .config import DataConfig, ModelConfig
from .model import GPTModel
from .tokenizer import TokenizerSpec, build_tokenizer
from .utils import autocast_kwargs, detect_device, load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from one or more NanoGPT-Next checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path or glob pattern.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text. Falls back to config sample_prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16", "fp8"])
    parser.add_argument("--tokenizer-path", type=str, default="", help="Optional tokenizer override.")
    parser.add_argument("--tokenizer-backend", type=str, default="", help="Optional tokenizer backend override.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output text file for single-checkpoint sampling.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for per-checkpoint sample outputs.")
    parser.add_argument("--export-jsonl", type=Path, default=None, help="Optional JSONL export path for sampled results.")
    parser.add_argument("--export-csv", type=Path, default=None, help="Optional CSV export path for sampled results.")
    return parser.parse_args()


def maybe_enable_float8(model: GPTModel, precision: str, device: torch.device) -> None:
    if precision != "fp8":
        return
    if device.type != "cuda":
        raise ValueError("FP8 checkpoint sampling requires a CUDA device.")
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    convert_to_float8_training(model, config=Float8LinearConfig())


def resolve_checkpoints(pattern: str) -> list[Path]:
    if any(char in pattern for char in "*?[]"):
        matches = [Path(match) for match in glob.glob(pattern)]
        if not matches:
            raise FileNotFoundError(f"No checkpoints matched pattern: {pattern}")
        return sorted(matches)
    checkpoint_path = Path(pattern)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return [checkpoint_path]


@torch.no_grad()
def compute_sequence_ppl(
    model: GPTModel,
    token_ids: list[int],
    device: torch.device,
    autocast_context,
) -> float | None:
    if len(token_ids) < 2:
        return None

    total_negative_log_likelihood = 0.0
    total_tokens = 0
    stride = model.config.max_seq_len
    for start in range(0, len(token_ids) - 1, stride):
        chunk = token_ids[start : start + stride + 1]
        if len(chunk) < 2:
            continue
        input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        with autocast_context():
            output = model(input_ids, targets)
        if output.main_loss is None:
            continue
        token_count = input_ids.size(1)
        total_negative_log_likelihood += float(output.main_loss.detach().cpu()) * token_count
        total_tokens += token_count

    if total_tokens == 0:
        return None
    mean_nll = total_negative_log_likelihood / total_tokens
    return math.exp(min(mean_nll, 20.0))


def sanitize_output_name(checkpoint_path: Path) -> str:
    raw = checkpoint_path.as_posix()
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return sanitized or checkpoint_path.stem


def write_result_exports(results: list[dict[str, Any]], jsonl_path: Path | None, csv_path: Path | None) -> None:
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "checkpoint",
            "step",
            "prompt",
            "text",
            "ppl",
            "max_new_tokens",
            "temperature",
            "top_k",
            "device",
            "precision",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({field: result.get(field, "") for field in fieldnames})


def sample_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config")
    if not isinstance(config_dict, dict):
        raise ValueError(f"Checkpoint does not contain embedded config metadata: {checkpoint_path}")

    model_config = ModelConfig(**config_dict["model"])
    data_config = DataConfig(**config_dict.get("data", {}))
    trainer_config = config_dict.get("trainer", {})

    tokenizer_spec = TokenizerSpec(
        backend=args.tokenizer_backend or data_config.tokenizer_backend,
        path=args.tokenizer_path or data_config.tokenizer_path,
    )
    tokenizer = build_tokenizer(tokenizer_spec)

    precision = trainer_config.get("precision", "bf16") if args.precision == "auto" else args.precision
    enable_autocast, autocast_dtype = autocast_kwargs(device, precision)
    if precision == "fp8":
        enable_autocast = device.type == "cuda"
        autocast_dtype = torch.bfloat16

    def autocast_context():
        if enable_autocast and autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    model = GPTModel(model_config)
    maybe_enable_float8(model, precision, device)
    load_model_state_dict(model, checkpoint["model"])
    model = model.to(device)
    model.eval()

    prompt = args.prompt or trainer_config.get("sample_prompt", "")
    if not prompt:
        raise ValueError(f"No prompt provided for checkpoint: {checkpoint_path}")

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError(f"Prompt encoded to zero tokens for checkpoint: {checkpoint_path}")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    top_k = None if args.top_k <= 0 else args.top_k
    with torch.no_grad():
        with autocast_context():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )
    output_token_ids = output_ids[0].tolist()
    text = tokenizer.decode(output_token_ids)
    ppl = compute_sequence_ppl(model, output_token_ids, device, autocast_context)

    return {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint.get("step", -1)),
        "prompt": prompt,
        "text": text,
        "ppl": ppl,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": top_k,
        "device": str(device),
        "precision": precision,
    }


def main() -> None:
    load_env_file()
    args = parse_args()
    checkpoint_paths = resolve_checkpoints(args.checkpoint)
    if len(checkpoint_paths) > 1 and args.output is not None:
        raise ValueError("--output only supports a single checkpoint. Use --output-dir for batch sampling.")

    device = detect_device(args.device)
    results = [sample_checkpoint(checkpoint_path, args, device) for checkpoint_path in checkpoint_paths]

    for result in results:
        ppl_text = "n/a" if result["ppl"] is None else f"{result['ppl']:.4f}"
        print(f"[{Path(result['checkpoint']).name}] step={result['step']} ppl={ppl_text}")
        print(result["text"])
        print()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(results[0]["text"], encoding="utf-8")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            checkpoint_path = Path(result["checkpoint"])
            output_name = sanitize_output_name(checkpoint_path)
            output_path = args.output_dir / f"{output_name}.txt"
            payload = f"checkpoint: {result['checkpoint']}\nstep: {result['step']}\nppl: {result['ppl']}\n\n{result['text']}"
            output_path.write_text(payload, encoding="utf-8")

    write_result_exports(results, args.export_jsonl, args.export_csv)


if __name__ == "__main__":
    main()
