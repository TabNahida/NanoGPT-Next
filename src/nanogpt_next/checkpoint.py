from __future__ import annotations

import os
import re
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch


def checkpoint_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_torch_save(payload: dict[str, Any], target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.{uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target_path


def _rotate_last_to_latest(ckpt_dir: Path) -> None:
    last_path = ckpt_dir / "last.pt"
    latest_path = ckpt_dir / "latest.pt"
    if last_path.exists():
        os.replace(last_path, latest_path)


def _link_or_copy(src_path: Path, dst_path: Path) -> Path:
    tmp_path = dst_path.with_name(f".{dst_path.name}.{uuid4().hex}.tmp")
    try:
        try:
            os.link(src_path, tmp_path)
        except OSError:
            shutil.copyfile(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return dst_path


def clone_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, OrderedDict):
        return OrderedDict((key, clone_to_cpu(item)) for key, item in value.items())
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return value


def unwrap_model(model: Any) -> Any:
    return getattr(model, "_orig_mod", model)


def export_model_state_dict(model: Any) -> OrderedDict[str, Any]:
    return unwrap_model(model).state_dict()


def load_model_state_dict(model: Any, state_dict: dict[str, Any]) -> None:
    target_model = unwrap_model(model)
    try:
        target_model.load_state_dict(state_dict)
        return
    except RuntimeError as first_error:
        normalized_state_dict = _strip_prefix_from_state_dict(state_dict, "_orig_mod.")
        if normalized_state_dict is state_dict:
            raise first_error
        target_model.load_state_dict(normalized_state_dict)


def _strip_prefix_from_state_dict(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    keys = list(state_dict.keys())
    if not keys or not all(isinstance(key, str) and key.startswith(prefix) for key in keys):
        return state_dict
    return OrderedDict((key[len(prefix) :], value) for key, value in state_dict.items())


def save_checkpoint(run_dir: str | Path, step: int, payload: dict[str, Any], keep_last_n: int) -> Path:
    ckpt_dir = checkpoint_dir(run_dir)
    ckpt_path = ckpt_dir / f"step-{step:08d}.pt"
    _atomic_torch_save(payload, ckpt_path)
    prune_checkpoints(ckpt_dir, keep_last_n)
    return ckpt_path


def save_last_checkpoint(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    ckpt_dir = checkpoint_dir(run_dir)
    staging_path = ckpt_dir / ".last.staging.pt"
    last_path = ckpt_dir / "last.pt"
    _atomic_torch_save(payload, staging_path)
    _rotate_last_to_latest(ckpt_dir)
    os.replace(staging_path, last_path)
    return last_path


def promote_checkpoint_to_last(run_dir: str | Path, checkpoint_path: str | Path) -> Path:
    ckpt_dir = checkpoint_dir(run_dir)
    source_path = Path(checkpoint_path)
    last_path = ckpt_dir / "last.pt"
    _rotate_last_to_latest(ckpt_dir)
    return _link_or_copy(source_path, last_path)


class AsyncLastCheckpointSaver:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self._condition = threading.Condition()
        self._pending_payload: dict[str, Any] | None = None
        self._active = False
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="nanogpt-next-last-checkpoint",
            daemon=True,
        )
        self._thread.start()

    def has_pending_work(self) -> bool:
        with self._condition:
            return self._active or self._pending_payload is not None

    def submit(self, payload: dict[str, Any]) -> None:
        with self._condition:
            self._raise_if_error_locked()
            if self._closed:
                raise RuntimeError("Cannot submit checkpoint after saver has been closed.")
            self._pending_payload = payload
            self._condition.notify_all()

    def flush(self) -> None:
        with self._condition:
            while (self._active or self._pending_payload is not None) and self._error is None:
                self._condition.wait()
            self._raise_if_error_locked()

    def close(self) -> None:
        self.flush()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Async last-checkpoint saver failed.") from self._error

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._pending_payload is None:
                    self._condition.wait()
                if self._closed and self._pending_payload is None:
                    return
                payload = self._pending_payload
                self._pending_payload = None
                self._active = True

            try:
                if payload is not None:
                    save_last_checkpoint(self.run_dir, payload)
            except BaseException as exc:
                with self._condition:
                    self._error = exc
            finally:
                with self._condition:
                    self._active = False
                    self._condition.notify_all()

    def _raise_if_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("Async last-checkpoint saver failed.") from self._error


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
    best_path: Path | None = None
    best_step = -1

    for candidate in (ckpt_dir / "last.pt", ckpt_dir / "latest.pt"):
        step = _checkpoint_step(candidate)
        if step > best_step:
            best_step = step
            best_path = candidate

    for candidate in ckpt_dir.glob("step-*.pt"):
        match = re.match(r"step-(\d{8})\.pt", candidate.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = candidate
    return best_path


def _checkpoint_step(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return -1
    try:
        return int(checkpoint.get("step", -1))
    except Exception:
        return -1


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
