from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nanogpt_next.utils.env import load_local_env


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    name: str
    warmup_ratio: float
    min_lr_ratio: float


@dataclass
class CheckpointConfig:
    save_top_k: int
    every_n_train_steps: int
    monitor: str
    mode: str


@dataclass
class TrainConfig:
    seed: int
    precision: str
    accelerator: str
    devices: Any
    num_nodes: int
    grad_accum_steps: int
    max_steps: int
    log_every_n_steps: int
    val_check_interval: int
    gradient_clip_val: float
    enable_progress_bar: bool
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    checkpoint: CheckpointConfig


@dataclass
class DataConfig:
    name: str
    train_path: str
    val_path: str | None
    text_column: str
    tokenizer_dir: str
    sequence_length: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle_shards: bool
    seed: int
    max_train_sequences: int | None
    max_val_sequences: int | None


@dataclass
class ModelConfig:
    name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    ffn_multiplier: float
    max_seq_len: int
    rope_theta: float
    partial_rotary_factor: float
    dropout: float
    rms_norm_eps: float
    tie_embeddings: bool
    mtp_num_future_tokens: int
    mtp_lambda: float
    use_activation_checkpointing: bool


@dataclass
class ExperimentConfig:
    name: str
    output_root: str
    notes: str | None
    model: ModelConfig
    data: DataConfig
    train: TrainConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _parse_overrides(overrides: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, value = item.split("=", 1)
        value_obj = yaml.safe_load(value)
        cursor = parsed
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value_obj
    return parsed


def _resolve_env_placeholders(cfg: Any) -> Any:
    if isinstance(cfg, str) and cfg.startswith("${env:") and cfg.endswith("}"):
        key = cfg[6:-1]
        if key not in os.environ:
            raise KeyError(f"Environment variable '{key}' is required by config")
        return os.environ[key]
    if isinstance(cfg, dict):
        return {k: _resolve_env_placeholders(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_resolve_env_placeholders(x) for x in cfg]
    return cfg


def _as_optimizer(cfg: dict[str, Any]) -> OptimizerConfig:
    return OptimizerConfig(
        name=cfg["name"],
        lr=float(cfg["lr"]),
        betas=tuple(cfg["betas"]),
        eps=float(cfg["eps"]),
        weight_decay=float(cfg["weight_decay"]),
    )


def _as_scheduler(cfg: dict[str, Any]) -> SchedulerConfig:
    return SchedulerConfig(
        name=cfg["name"],
        warmup_ratio=float(cfg["warmup_ratio"]),
        min_lr_ratio=float(cfg["min_lr_ratio"]),
    )


def _as_checkpoint(cfg: dict[str, Any]) -> CheckpointConfig:
    return CheckpointConfig(
        save_top_k=int(cfg["save_top_k"]),
        every_n_train_steps=int(cfg["every_n_train_steps"]),
        monitor=cfg["monitor"],
        mode=cfg["mode"],
    )


def _as_train(cfg: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        seed=int(cfg["seed"]),
        precision=cfg["precision"],
        accelerator=cfg["accelerator"],
        devices=cfg["devices"],
        num_nodes=int(cfg["num_nodes"]),
        grad_accum_steps=int(cfg["grad_accum_steps"]),
        max_steps=int(cfg["max_steps"]),
        log_every_n_steps=int(cfg["log_every_n_steps"]),
        val_check_interval=int(cfg["val_check_interval"]),
        gradient_clip_val=float(cfg["gradient_clip_val"]),
        enable_progress_bar=bool(cfg.get("enable_progress_bar", False)),
        optimizer=_as_optimizer(cfg["optimizer"]),
        scheduler=_as_scheduler(cfg["scheduler"]),
        checkpoint=_as_checkpoint(cfg["checkpoint"]),
    )


def _as_data(cfg: dict[str, Any]) -> DataConfig:
    return DataConfig(
        name=cfg["name"],
        train_path=cfg["train_path"],
        val_path=cfg.get("val_path"),
        text_column=cfg["text_column"],
        tokenizer_dir=cfg["tokenizer_dir"],
        sequence_length=int(cfg["sequence_length"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        shuffle_shards=bool(cfg["shuffle_shards"]),
        seed=int(cfg["seed"]),
        max_train_sequences=cfg.get("max_train_sequences"),
        max_val_sequences=cfg.get("max_val_sequences"),
    )


def _as_model(cfg: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name=cfg["name"],
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        num_kv_heads=int(cfg["num_kv_heads"]),
        ffn_multiplier=float(cfg["ffn_multiplier"]),
        max_seq_len=int(cfg["max_seq_len"]),
        rope_theta=float(cfg["rope_theta"]),
        partial_rotary_factor=float(cfg["partial_rotary_factor"]),
        dropout=float(cfg["dropout"]),
        rms_norm_eps=float(cfg["rms_norm_eps"]),
        tie_embeddings=bool(cfg["tie_embeddings"]),
        mtp_num_future_tokens=int(cfg["mtp_num_future_tokens"]),
        mtp_lambda=float(cfg["mtp_lambda"]),
        use_activation_checkpointing=bool(cfg["use_activation_checkpointing"]),
    )


def load_experiment_config(
    experiment_path: str | Path,
    overrides: list[str] | None = None,
    project_root: str | Path | None = None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    overrides = overrides or []
    exp_path = Path(experiment_path).resolve()
    root = Path(project_root).resolve() if project_root else exp_path.parents[2]
    load_local_env(root)

    exp_cfg = _read_yaml(exp_path)
    defaults = exp_cfg.get("defaults", {})
    model_cfg = _read_yaml((root / "configs" / defaults["model"]).resolve())
    data_cfg = _read_yaml((root / "configs" / defaults["data"]).resolve())
    train_cfg = _read_yaml((root / "configs" / defaults["train"]).resolve())

    merged: dict[str, Any] = {
        "name": exp_cfg["name"],
        "output_root": exp_cfg.get("output_root", "runs"),
        "notes": exp_cfg.get("notes"),
        "model": _merge_dict(model_cfg, exp_cfg.get("model", {})),
        "data": _merge_dict(data_cfg, exp_cfg.get("data", {})),
        "train": _merge_dict(train_cfg, exp_cfg.get("train", {})),
    }
    merged = _merge_dict(merged, _parse_overrides(overrides))
    merged = _resolve_env_placeholders(merged)

    config = ExperimentConfig(
        name=merged["name"],
        output_root=merged["output_root"],
        notes=merged.get("notes"),
        model=_as_model(merged["model"]),
        data=_as_data(merged["data"]),
        train=_as_train(merged["train"]),
    )
    return config, merged


def save_resolved_config(resolved_cfg: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=False)
