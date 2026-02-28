from __future__ import annotations

from pathlib import Path

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl  # type: ignore

from torch.utils.data import DataLoader

from nanogpt_next.data.parquet_stream import StreamDatasetConfig, build_loader
from nanogpt_next.tokenizer.wrapper import TokenizerWrapper
from nanogpt_next.utils.config import DataConfig


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig, project_root: Path) -> None:
        super().__init__()
        self.cfg = cfg
        self.project_root = project_root
        self.tokenizer = TokenizerWrapper.from_dir(project_root / cfg.tokenizer_dir)

    def setup(self, stage: str | None = None) -> None:
        del stage

    def train_dataloader(self) -> DataLoader:
        stream_cfg = StreamDatasetConfig(
            path_pattern=self.cfg.train_path,
            text_column=self.cfg.text_column,
            sequence_length=self.cfg.sequence_length,
            seed=self.cfg.seed,
            shuffle_shards=self.cfg.shuffle_shards,
            max_sequences=self.cfg.max_train_sequences,
        )
        return build_loader(
            cfg=stream_cfg,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        val_path = self.cfg.val_path or self.cfg.train_path
        stream_cfg = StreamDatasetConfig(
            path_pattern=val_path,
            text_column=self.cfg.text_column,
            sequence_length=self.cfg.sequence_length,
            seed=self.cfg.seed + 1,
            shuffle_shards=False,
            max_sequences=self.cfg.max_val_sequences,
        )
        return build_loader(
            cfg=stream_cfg,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
