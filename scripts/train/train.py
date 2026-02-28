from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ModuleNotFoundError:
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # type: ignore
    from pytorch_lightning.loggers import CSVLogger  # type: ignore

from nanogpt_next.data.datamodule import PretrainDataModule
from nanogpt_next.train.lightning_module import NanoGPTLightningModule
from nanogpt_next.utils.config import load_experiment_config, save_resolved_config
from nanogpt_next.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NanoGPT-Next")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/model_a_1k.yaml",
        help="Experiment YAML path",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override, e.g. train.max_steps=1000",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    exp_cfg, resolved = load_experiment_config(
        args.config,
        args.override,
        project_root=project_root,
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = project_root / exp_cfg.output_root / exp_cfg.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(resolved, run_dir / "resolved_config.yaml")

    seed_everything(exp_cfg.train.seed)
    pl.seed_everything(exp_cfg.train.seed, workers=True)

    datamodule = PretrainDataModule(exp_cfg.data, project_root=project_root)
    # Always sync vocab size to tokenizer artifact.
    exp_cfg.model.vocab_size = datamodule.tokenizer.vocab_size
    model = NanoGPTLightningModule(exp_cfg.model, exp_cfg.train)

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="step{step:08d}-valloss{val/loss:.4f}",
        save_top_k=exp_cfg.train.checkpoint.save_top_k,
        every_n_train_steps=exp_cfg.train.checkpoint.every_n_train_steps,
        monitor=exp_cfg.train.checkpoint.monitor,
        mode=exp_cfg.train.checkpoint.mode,
        save_last=True,
        auto_insert_metric_name=False,
    )
    logger = CSVLogger(save_dir=str(run_dir), name="logs")

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        accelerator=exp_cfg.train.accelerator,
        devices=exp_cfg.train.devices,
        num_nodes=exp_cfg.train.num_nodes,
        precision=exp_cfg.train.precision,
        max_steps=exp_cfg.train.max_steps,
        accumulate_grad_batches=exp_cfg.train.grad_accum_steps,
        gradient_clip_val=exp_cfg.train.gradient_clip_val,
        log_every_n_steps=exp_cfg.train.log_every_n_steps,
        val_check_interval=exp_cfg.train.val_check_interval,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step")],
        logger=logger,
        deterministic=True,
        enable_progress_bar=exp_cfg.train.enable_progress_bar,
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
