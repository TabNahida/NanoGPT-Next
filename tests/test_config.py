from __future__ import annotations

from pathlib import Path

from nanogpt_next.utils.config import load_experiment_config


def test_load_experiment_config() -> None:
    cfg, resolved = load_experiment_config(
        experiment_path="configs/experiments/model_a_1k.yaml",
        overrides=["train.max_steps=123", "data.sequence_length=256"],
        project_root=Path.cwd(),
    )
    assert cfg.train.max_steps == 123
    assert cfg.data.sequence_length == 256
    assert resolved["train"]["max_steps"] == 123
