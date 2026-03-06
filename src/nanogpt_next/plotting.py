from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_metrics_csv(csv_path: str | Path, output_path: str | Path) -> None:
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    if not csv_path.exists():
        return

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        return

    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    if train_rows:
        steps = [int(row["step"]) for row in train_rows]
        axes[0].plot(steps, [float(row["loss"]) for row in train_rows], label="train")
        axes[0].plot(steps, [float(row["main_loss"]) for row in train_rows], label="main", alpha=0.7)
        axes[0].set_title("Training Loss")
        axes[0].legend()

        axes[1].plot(steps, [float(row["lr"]) for row in train_rows], color="tab:orange")
        axes[1].set_title("Learning Rate")

        axes[2].plot(steps, [float(row["tokens_per_sec"]) for row in train_rows], color="tab:green")
        axes[2].set_title("Tokens / Sec")

        axes[3].plot(steps, [float(row["grad_norm"]) for row in train_rows], color="tab:red")
        axes[3].set_title("Grad Norm")

    if val_rows:
        val_steps = [int(row["step"]) for row in val_rows]
        axes[0].plot(val_steps, [float(row["loss"]) for row in val_rows], label="val", linestyle="--")
        axes[0].legend()

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.set_xlabel("Step")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NanoGPT-Next training metrics.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("metrics.png"))
    args = parser.parse_args()
    plot_metrics_csv(args.csv_path, args.output)


if __name__ == "__main__":
    main()
