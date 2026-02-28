from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from glob import glob

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

from nanogpt_next.tokenizer.wrapper import TokenizerWrapper
from nanogpt_next.utils.seed import seed_worker


@dataclass
class StreamDatasetConfig:
    path_pattern: str
    text_column: str
    sequence_length: int
    seed: int
    shuffle_shards: bool
    max_sequences: int | None


class PackedParquetDataset(IterableDataset[torch.Tensor]):
    def __init__(
        self,
        cfg: StreamDatasetConfig,
        tokenizer: TokenizerWrapper,
        add_eos: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.add_eos = add_eos

        self.shards = sorted(glob(cfg.path_pattern))
        if not self.shards:
            raise FileNotFoundError(f"No parquet shards matched pattern: {cfg.path_pattern}")

    def _iter_shards(self) -> list[str]:
        shards = list(self.shards)
        if self.cfg.shuffle_shards:
            rng = random.Random(self.cfg.seed)
            rng.shuffle(shards)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return shards
        return shards[worker_info.id :: worker_info.num_workers]

    def _pick_text_column(self, schema: pa.Schema) -> str:
        if self.cfg.text_column in schema.names:
            return self.cfg.text_column
        for field in schema:
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                return field.name
        raise ValueError(
            f"Configured text column '{self.cfg.text_column}' not found and no string column exists"
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        seq_len = self.cfg.sequence_length
        emitted = 0
        buffer: list[int] = []
        eos_id = self.tokenizer.eos_id if self.add_eos else None

        for shard in self._iter_shards():
            parquet_file = pq.ParquetFile(shard)
            text_col = self._pick_text_column(parquet_file.schema_arrow)

            for batch in parquet_file.iter_batches(columns=[text_col], batch_size=256):
                texts = batch[text_col].to_pylist()
                tokenized_docs = self.tokenizer.encode_batch(texts, add_special_tokens=False)
                for ids in tokenized_docs:
                    buffer.extend(ids)
                    if eos_id is not None:
                        buffer.append(eos_id)

                    while len(buffer) >= seq_len:
                        sample = torch.tensor(buffer[:seq_len], dtype=torch.long)
                        buffer = buffer[seq_len:]
                        yield sample
                        emitted += 1
                        if self.cfg.max_sequences is not None and emitted >= self.cfg.max_sequences:
                            return


def build_loader(
    cfg: StreamDatasetConfig,
    tokenizer: TokenizerWrapper,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = PackedParquetDataset(cfg=cfg, tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker,
    )
