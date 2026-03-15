from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden_dim: int
    dropout: float = 0.0
    rope_base: float = 10000.0
    rope_pct: float = 0.5
    mtp_heads: int = 0
    mtp_weight: float = 0.0
    mtp_decay: float = 1.0
    tie_word_embeddings: bool = True
    use_bias: bool = False
    initializer_std: float = 0.02

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA.")
        if self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads cannot exceed n_heads.")
        if not 0.0 < self.rope_pct <= 1.0:
            raise ValueError("rope_pct must be in (0, 1].")
        if self.mtp_heads < 0:
            raise ValueError("mtp_heads cannot be negative.")


@dataclass(slots=True)
class DataConfig:
    train_glob: str = ""
    val_glob: str = ""
    auto_val_split: bool = False
    auto_val_ratio: float = 0.01
    auto_val_seed: int = 1337
    tokenizer_backend: str = "auto"
    tokenizer_path: str = "Tokenizer/V1/tokenizer.json"
    text_column: str = "auto"
    candidate_text_columns: list[str] = field(
        default_factory=lambda: ["text", "plain_text", "content", "document"]
    )
    add_bos: bool = False
    add_eos: bool = True
    shuffle_files: bool = True
    row_batch_size: int = 128
    num_workers: int = 0


@dataclass(slots=True)
class OptimizerConfig:
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.1
    adamw_fused: bool = True
    muon_lr: float = 3e-4
    muon_weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"


@dataclass(slots=True)
class TrainerConfig:
    device: str = "auto"
    precision: str = "bf16"
    matmul_precision: str = "high"
    compile_model: bool = False
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 32
    max_steps: int = 1000
    warmup_steps: int = 100
    lr_decay_steps: int = 1000
    min_lr_ratio: float = 0.1
    clip_grad_norm: float = 1.0
    log_every: int = 20
    eval_every: int = 0
    eval_batches: int = 0
    save_every: int = 1000
    keep_last_n_checkpoints: int = 0
    resume_from: str = "auto"
    output_dir: str = "outputs"
    sample_every: int = 0
    sample_tokens: int = 128
    sample_prompt: str = ""


@dataclass(slots=True)
class LoggingConfig:
    enable_csv: bool = True
    enable_jsonl: bool = True
    enable_tensorboard: bool = True
    enable_plot: bool = True
    plot_every: int = 100


@dataclass(slots=True)
class MonitoringConfig:
    thermal_pause_enabled: bool = False
    thermal_max_celsius: float = 83.0
    thermal_resume_celsius: float = 75.0
    thermal_poll_interval: float = 15.0
    thermal_query_command: str = ""
    status_api_enabled: bool = False
    status_api_host: str = "127.0.0.1"
    status_api_port: int = 8008


@dataclass(slots=True)
class ExperimentConfig:
    run_name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig

    def validate(self) -> None:
        self.model.validate()
        if self.data.auto_val_split and not 0.0 < self.data.auto_val_ratio < 1.0:
            raise ValueError("data.auto_val_ratio must be in (0, 1) when auto_val_split is enabled.")
        if self.trainer.precision not in {"fp32", "fp16", "bf16", "fp8"}:
            raise ValueError("precision must be one of fp32, fp16, bf16, fp8.")
        if self.trainer.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        if self.trainer.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.trainer.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.trainer.log_every <= 0:
            raise ValueError("log_every must be positive.")
        if self.monitoring.thermal_pause_enabled:
            if self.monitoring.thermal_poll_interval <= 0:
                raise ValueError("monitoring.thermal_poll_interval must be positive when thermal pause is enabled.")
            if self.monitoring.thermal_resume_celsius > self.monitoring.thermal_max_celsius:
                raise ValueError(
                    "monitoring.thermal_resume_celsius must be less than or equal to monitoring.thermal_max_celsius."
                )
        if self.monitoring.status_api_enabled and self.monitoring.status_api_port <= 0:
            raise ValueError("monitoring.status_api_port must be positive when the status API is enabled.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_dict(config_dict: dict[str, Any]) -> ExperimentConfig:
    experiment = ExperimentConfig(
        run_name=config_dict["run_name"],
        seed=config_dict.get("seed", 1337),
        model=ModelConfig(**config_dict["model"]),
        data=DataConfig(**config_dict.get("data", {})),
        optimizer=OptimizerConfig(**config_dict.get("optimizer", {})),
        trainer=TrainerConfig(**config_dict.get("trainer", {})),
        logging=LoggingConfig(**config_dict.get("logging", {})),
        monitoring=MonitoringConfig(**config_dict.get("monitoring", {})),
    )
    experiment.validate()
    return experiment


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return _from_dict(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
