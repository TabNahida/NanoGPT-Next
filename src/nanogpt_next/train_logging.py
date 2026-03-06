from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import LoggingConfig
from .plotting import plot_metrics_csv


class MetricsLogger:
    def __init__(self, run_dir: str | Path, config: LoggingConfig) -> None:
        self.run_dir = Path(run_dir)
        self.config = config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.plot_path = self.run_dir / "metrics.png"
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._fieldnames = [
            "step",
            "split",
            "loss",
            "main_loss",
            "mtp_loss",
            "lr",
            "grad_norm",
            "tokens_seen",
            "tokens_per_sec",
            "ppl",
        ]
        self._tb = None
        self._last_plot_step = -1

        if config.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except Exception:
                self._tb = None

    def log(self, step: int, split: str, metrics: dict[str, Any]) -> None:
        row = {"step": step, "split": split}
        row.update(metrics)
        if self.config.enable_csv:
            self._write_csv(row)
        if self.config.enable_jsonl:
            self._write_jsonl(row)
        if self._tb is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb.add_scalar(f"{split}/{key}", value, step)
        if self.config.enable_plot and split == "train":
            if self._last_plot_step < 0 or step - self._last_plot_step >= self.config.plot_every:
                plot_metrics_csv(self.csv_path, self.plot_path)
                self._last_plot_step = step

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self._tb is not None:
            self._tb.add_text(tag, text, step)
        text_dir = self.run_dir / "samples"
        text_dir.mkdir(parents=True, exist_ok=True)
        (text_dir / f"{tag}-{step:08d}.txt").write_text(text, encoding="utf-8")

    def close(self) -> None:
        if self.config.enable_plot:
            plot_metrics_csv(self.csv_path, self.plot_path)
        if self._tb is not None:
            self._tb.close()

    def _write_csv(self, row: dict[str, Any]) -> None:
        payload = {field: row.get(field, "") for field in self._fieldnames}
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(payload)

    def _write_jsonl(self, row: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
