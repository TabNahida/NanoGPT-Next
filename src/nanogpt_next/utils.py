from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def detect_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def autocast_kwargs(device: torch.device, precision: str) -> tuple[bool, torch.dtype | None]:
    if precision == "bf16":
        return (device.type in {"cuda", "xpu", "cpu"}, torch.bfloat16)
    if precision == "fp16":
        return (device.type == "cuda", torch.float16)
    return (False, None)


def cosine_lr_multiplier(step: int, warmup_steps: int, decay_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if step >= decay_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def format_num(num: float) -> str:
    abs_num = abs(num)
    if abs_num >= 1e9:
        return f"{num / 1e9:.2f}B"
    if abs_num >= 1e6:
        return f"{num / 1e6:.2f}M"
    if abs_num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return f"{num:.0f}"
