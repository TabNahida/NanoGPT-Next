from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import torch


def checkpoint_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(run_dir: str | Path, step: int, payload: dict[str, Any], keep_last_n: int) -> Path:
    ckpt_dir = checkpoint_dir(run_dir)
    ckpt_path = ckpt_dir / f"step-{step:08d}.pt"
    torch.save(payload, ckpt_path)
    prune_checkpoints(ckpt_dir, keep_last_n)
    return ckpt_path


def save_last_checkpoint(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    ckpt_dir = checkpoint_dir(run_dir)
    last_path = ckpt_dir / "last.pt"
    torch.save(payload, last_path)
    latest_path = ckpt_dir / "latest.pt"
    shutil.copyfile(last_path, latest_path)
    return last_path


def prune_checkpoints(ckpt_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    step_files = sorted(
        (
            path
            for path in ckpt_dir.glob("step-*.pt")
            if re.match(r"step-\d{8}\.pt", path.name)
        ),
        key=lambda path: path.name,
    )
    for stale in step_files[:-keep_last_n]:
        stale.unlink(missing_ok=True)


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    ckpt_dir = checkpoint_dir(run_dir)
    last_path = ckpt_dir / "last.pt"
    if last_path.exists():
        return last_path
    latest_path = ckpt_dir / "latest.pt"
    if latest_path.exists():
        return latest_path
    return None


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
