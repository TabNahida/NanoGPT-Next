from __future__ import annotations

import glob
import hashlib
import os
import random

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .config import DataConfig
from .tokenizer import TokenizerAdapter, TokenizerSpec, build_tokenizer


def resolve_data_glob(raw_glob: str, env_keys: tuple[str, ...]) -> str:
    if raw_glob:
        return raw_glob
    for key in env_keys:
        env_value = os.environ.get(key)
        if env_value:
            return env_value
    return ""


def resolve_text_column(files: list[str], config: DataConfig) -> str:
    if not files:
        raise FileNotFoundError("No parquet files matched the configured glob.")
    schema = pq.ParquetFile(files[0]).schema_arrow
    column_names = set(schema.names)
    if config.text_column != "auto":
        if config.text_column not in column_names:
            raise ValueError(
                f"Configured text column '{config.text_column}' not found. Available columns: {sorted(column_names)}"
            )
        return config.text_column
    for candidate in config.candidate_text_columns:
        if candidate in column_names:
            return candidate
    raise ValueError(
        f"Unable to auto-detect text column. Available columns: {sorted(column_names)}"
    )


class PackedParquetDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        files: list[str],
        tokenizer_spec: TokenizerSpec,
        text_column: str,
        seq_len: int,
        row_batch_size: int,
        add_bos: bool,
        add_eos: bool,
        shuffle_files: bool,
        repeat: bool,
        seed: int,
        split_name: str = "all",
        auto_val_ratio: float = 0.0,
        auto_val_seed: int = 1337,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.tokenizer_spec = tokenizer_spec
        self.text_column = text_column
        self.seq_len = seq_len
        self.row_batch_size = row_batch_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.shuffle_files = shuffle_files
        self.repeat = repeat
        self.seed = seed
        self.split_name = split_name
        self.auto_val_ratio = auto_val_ratio
        self.auto_val_seed = auto_val_seed

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        files = self.files
        if not files:
            raise FileNotFoundError("No parquet files were provided to the dataset.")
        worker_files = files[worker_id::num_workers]
        if not worker_files:
            return iter(())

        tokenizer = build_tokenizer(self.tokenizer_spec)
        rng = random.Random(self.seed + worker_id)
        epoch = 0

        while True:
            epoch_files = list(worker_files)
            if self.shuffle_files:
                rng.shuffle(epoch_files)

            buffer: list[int] = []
            buffer_start = 0
            for file_path in epoch_files:
                parquet_file = pq.ParquetFile(file_path)
                row_offset = 0
                for batch in parquet_file.iter_batches(
                    batch_size=self.row_batch_size,
                    columns=[self.text_column],
                    use_threads=True,
                ):
                    values = batch.column(0).to_pylist()
                    for local_row_index, text in enumerate(values):
                        row_index = row_offset + local_row_index
                        if not self._include_row(file_path, row_index):
                            continue
                        if not text:
                            continue
                        token_ids = tokenizer.encode(str(text))
                        if self.add_bos and tokenizer.bos_token_id is not None:
                            token_ids = [tokenizer.bos_token_id] + token_ids
                        if self.add_eos and tokenizer.eos_token_id is not None:
                            token_ids = token_ids + [tokenizer.eos_token_id]
                        buffer.extend(token_ids)
                        while len(buffer) - buffer_start >= self.seq_len + 1:
                            chunk = buffer[buffer_start : buffer_start + self.seq_len + 1]
                            buffer_start += self.seq_len + 1
                            x = torch.tensor(chunk[:-1], dtype=torch.long)
                            y = torch.tensor(chunk[1:], dtype=torch.long)
                            yield x, y
                        if buffer_start > 16384:
                            buffer = buffer[buffer_start:]
                            buffer_start = 0
                    row_offset += len(values)

            if not self.repeat:
                return
            epoch += 1
            rng.seed(self.seed + worker_id + epoch)

    def _include_row(self, file_path: str, row_index: int) -> bool:
        if self.split_name == "all":
            return True
        key = f"{self.auto_val_seed}:{file_path}:{row_index}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        fraction = int.from_bytes(digest, "big") / float(1 << 64)
        is_val = fraction < self.auto_val_ratio
        if self.split_name == "val":
            return is_val
        if self.split_name == "train":
            return not is_val
        raise ValueError(f"Unknown split name: {self.split_name}")


def build_tokenizer_and_dataloaders(
    data_config: DataConfig,
    seq_len: int,
    seed: int,
    micro_batch_size: int,
) -> tuple[TokenizerAdapter, DataLoader, DataLoader | None]:
    train_glob = resolve_data_glob(data_config.train_glob, ("DATA_PATH",))
    if not train_glob:
        raise ValueError("No train_glob configured and DATA_PATH is not set.")

    tokenizer_spec = TokenizerSpec(
        backend=data_config.tokenizer_backend,
        path=data_config.tokenizer_path,
    )
    tokenizer = build_tokenizer(tokenizer_spec)

    train_files = sorted(glob.glob(train_glob))
    text_column = resolve_text_column(train_files, data_config)
    use_auto_val_split = bool(data_config.auto_val_split and not data_config.val_glob)
    train_dataset = PackedParquetDataset(
        files=train_files,
        tokenizer_spec=tokenizer_spec,
        text_column=text_column,
        seq_len=seq_len,
        row_batch_size=data_config.row_batch_size,
        add_bos=data_config.add_bos,
        add_eos=data_config.add_eos,
        shuffle_files=data_config.shuffle_files,
        repeat=True,
        seed=seed,
        split_name="train" if use_auto_val_split else "all",
        auto_val_ratio=data_config.auto_val_ratio,
        auto_val_seed=data_config.auto_val_seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        num_workers=data_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader: DataLoader | None = None
    if data_config.val_glob:
        val_files = sorted(glob.glob(data_config.val_glob))
        val_text_column = resolve_text_column(val_files, data_config)
        val_dataset = PackedParquetDataset(
            files=val_files,
            tokenizer_spec=tokenizer_spec,
            text_column=val_text_column,
            seq_len=seq_len,
            row_batch_size=data_config.row_batch_size,
            add_bos=data_config.add_bos,
            add_eos=data_config.add_eos,
            shuffle_files=False,
            repeat=False,
            seed=seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=micro_batch_size,
            num_workers=data_config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    elif use_auto_val_split:
        val_dataset = PackedParquetDataset(
            files=train_files,
            tokenizer_spec=tokenizer_spec,
            text_column=text_column,
            seq_len=seq_len,
            row_batch_size=data_config.row_batch_size,
            add_bos=data_config.add_bos,
            add_eos=data_config.add_eos,
            shuffle_files=False,
            repeat=False,
            seed=seed,
            split_name="val",
            auto_val_ratio=data_config.auto_val_ratio,
            auto_val_seed=data_config.auto_val_seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=micro_batch_size,
            num_workers=data_config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    return tokenizer, train_loader, val_loader
