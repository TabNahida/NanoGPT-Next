# NanoGPT-Next

A minimal PyTorch training stack for sub-1B language-model experiments, with a training path built around `RMSNorm + SwiGLU + partial RoPE + GQA + MTP + Muon/AdamW`.

## Features

- Model architecture:
  - `RMSNorm`
  - `SwiGLU`
  - `RoPE` with partial rotary ratio control
  - `GQA` grouped-query attention
  - optional `MTP` auxiliary prediction heads
- Training and runtime:
  - `PyTorch` training on `CUDA`, `XPU`, and CPU fallback
  - default `BF16`
  - optional experimental `FP8` training on CUDA via `torchao`
  - optional `torch.compile`
- Data pipeline:
  - parquet streaming with `pyarrow`
  - tokenizer backend prefers `TokenFlux` when installed
  - automatic fallback to bundled `Tokenizer/V1/tokenizer.json`
  - optional automatic validation split from the training parquet stream
- Training tooling:
  - periodic `step-XXXXXXXX.pt` checkpoints
  - rolling `last.pt` and `latest.pt`
  - asynchronous `last.pt` refresh on log steps
  - `auto resume` support
  - checkpoint compatibility across compiled and uncompiled models
  - CSV / JSONL / TensorBoard / PNG logging
  - optional online text sampling during training
- Long-run operations support:
  - GPU temperature throttling on log steps
  - optional local REST status API for progress, speed, losses, checkpoints, and thermal state
- Offline inference tooling:
  - single-checkpoint sampling
  - wildcard batch sampling from multiple checkpoints
  - JSONL / CSV export of sampled results
  - per-sample `PPL` calculation

## Provided Configs

- `configs/pretrain_300m_1k.json`
  - target class: about `300M`
  - context length: `1024`
- `configs/pretrain_300m_1k_fast.json`
  - fast pretrain variant for earlier-stage iteration
  - context length: `1024`
- `configs/pretrain_500m_4k.json`
  - target class: about `500M`
  - context length: `4096`
- `configs/smoke_tiny.json`
  - CPU-friendly smoke test

## Install

Base install:

```bash
python -m pip install -e .
```

Optional visualization support:

```bash
python -m pip install -e .[viz]
```

Notes:

- `tensorboard` is optional. If it is missing, training still works and falls back to `CSV + JSONL + PNG`.
- `torchao` is only required if you actually use `FP8`.
- automatic temperature monitoring currently supports `CUDA` through `nvidia-smi` by default.
- for `XPU` or custom environments, use `monitoring.thermal_query_command`.

## Data Format

The training loader expects parquet files with one text column such as:

- `text`
- `plain_text`
- `content`
- `document`

You can point training data to:

- `data.train_glob` in the config
- `DATA_PATH` in `.env`

Example `.env`:

```env
DATA_PATH=/OpenWebText/plain_text/train-*-of-00080.parquet
```

Validation can be configured in two ways:

- explicit validation set with `data.val_glob`
- automatic split from `data.train_glob` with:
  - `data.auto_val_split`
  - `data.auto_val_ratio`
  - `data.auto_val_seed`

Example:

```json
{
  "data": {
    "train_glob": "D:/data/train-*.parquet",
    "val_glob": "",
    "auto_val_split": true,
    "auto_val_ratio": 0.01,
    "auto_val_seed": 1337
  }
}
```

## Quick Start

Train:

```bash
python -m nanogpt_next.train --config configs/pretrain_300m_1k.json
```

Common overrides:

```bash
python -m nanogpt_next.train ^
  --config configs/pretrain_500m_4k.json ^
  --device cuda ^
  --precision bf16 ^
  --max-steps 2000
```

Run one checkpoint sample:

```bash
python -m nanogpt_next.sample ^
  --checkpoint outputs/pretrain-300m-1k/checkpoints/last.pt ^
  --prompt "The future of open language models is" ^
  --max-new-tokens 128
```

Batch-sample multiple checkpoints:

```bash
python -m nanogpt_next.sample ^
  --checkpoint "outputs/pretrain-300m-1k/checkpoints/step-*.pt" ^
  --prompt "The future of open language models is" ^
  --max-new-tokens 128 ^
  --output-dir outputs/pretrain-300m-1k/batch-samples ^
  --export-jsonl outputs/pretrain-300m-1k/batch-samples/results.jsonl ^
  --export-csv outputs/pretrain-300m-1k/batch-samples/results.csv
```

## Configuration Reference

Configuration is split into six top-level blocks:

- `model`
- `data`
- `optimizer`
- `trainer`
- `logging`
- `monitoring`

### `model`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `vocab_size` | `int` | Vocabulary size. This is auto-adjusted to the tokenizer vocab at runtime if needed. |
| `max_seq_len` | `int` | Maximum sequence length and generation context window. |
| `n_layers` | `int` | Number of transformer blocks. |
| `d_model` | `int` | Hidden size. Must be divisible by `n_heads`. |
| `n_heads` | `int` | Number of attention query heads. |
| `n_kv_heads` | `int` | Number of key/value heads for `GQA`. Must divide `n_heads`. |
| `ffn_hidden_dim` | `int` | Feed-forward hidden dimension used by `SwiGLU`. |
| `dropout` | `float` | Dropout probability used on token embeddings. |
| `rope_base` | `float` | RoPE base frequency. |
| `rope_pct` | `float` | Fraction of head dimension that uses rotary embedding. Must be in `(0, 1]`. |
| `mtp_heads` | `int` | Number of auxiliary `MTP` heads. `0` disables MTP. |
| `mtp_weight` | `float` | Weight for the first `MTP` loss term. |
| `mtp_decay` | `float` | Multiplicative decay applied to deeper `MTP` heads. |
| `tie_word_embeddings` | `bool` | Whether to tie the LM head to token embeddings. |
| `use_bias` | `bool` | Whether linear layers in the model use bias terms. |
| `initializer_std` | `float` | Base standard deviation for parameter initialization. |

### `data`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `train_glob` | `str` | Glob pattern for training parquet files. |
| `val_glob` | `str` | Glob pattern for validation parquet files. If empty and `auto_val_split=true`, validation is split from `train_glob`. |
| `auto_val_split` | `bool` | Enables deterministic row-level split from `train_glob` into train and validation streams. |
| `auto_val_ratio` | `float` | Validation fraction used when `auto_val_split=true`. Must be in `(0, 1)`. |
| `auto_val_seed` | `int` | Seed controlling deterministic row assignment for auto validation split. |
| `tokenizer_backend` | `str` | `auto`, `tokenflux`, or `hf`. `auto` prefers `TokenFlux`, then falls back to Hugging Face `tokenizers`. |
| `tokenizer_path` | `str` | Path to tokenizer JSON file. |
| `text_column` | `str` | Explicit parquet text column. Use `auto` to probe candidate columns. |
| `candidate_text_columns` | `list[str]` | Candidate column names checked when `text_column=auto`. |
| `add_bos` | `bool` | Prepend BOS token if the tokenizer exposes one. |
| `add_eos` | `bool` | Append EOS token if the tokenizer exposes one. |
| `shuffle_files` | `bool` | Shuffle parquet file order before streaming. |
| `row_batch_size` | `int` | Number of parquet rows loaded per reader batch before tokenization. |
| `num_workers` | `int` | PyTorch DataLoader worker count. |

### `optimizer`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `adamw_lr` | `float` | Base learning rate for `AdamW`. |
| `adamw_betas` | `tuple[float, float]` | `AdamW` beta coefficients. |
| `adamw_eps` | `float` | `AdamW` epsilon. |
| `adamw_weight_decay` | `float` | Weight decay for `AdamW` decay parameter group. |
| `adamw_fused` | `bool` | Enables fused `AdamW` when available on CUDA. |
| `muon_lr` | `float` | Base learning rate for `Muon`. |
| `muon_weight_decay` | `float` | Weight decay for `Muon`. |
| `muon_momentum` | `float` | Momentum factor for `Muon`. |
| `muon_nesterov` | `bool` | Enables Nesterov momentum in `Muon`. |
| `muon_ns_steps` | `int` | Number of Newton-Schulz steps for `Muon`. |
| `muon_adjust_lr_fn` | `str` | Name of the `Muon` LR adjustment strategy. |

### `trainer`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `device` | `str` | `auto`, `cpu`, `cuda`, or `xpu`. |
| `precision` | `str` | `fp32`, `fp16`, `bf16`, or `fp8`. `fp8` is CUDA-only. |
| `matmul_precision` | `str` | PyTorch float32 matmul precision mode, usually `high` or `highest`. |
| `compile_model` | `bool` | Enables `torch.compile`. Checkpoint save/load remains compatible with both compiled and uncompiled models. |
| `micro_batch_size` | `int` | Per-forward micro batch size. |
| `gradient_accumulation_steps` | `int` | Number of micro steps before one optimizer step. |
| `max_steps` | `int` | Total optimizer steps to run. |
| `warmup_steps` | `int` | LR warmup steps. |
| `lr_decay_steps` | `int` | Total steps used by cosine LR decay. |
| `min_lr_ratio` | `float` | Minimum cosine-decayed LR expressed as a ratio of the base LR. |
| `clip_grad_norm` | `float` | Global gradient clipping max norm. |
| `log_every` | `int` | Training metrics logging interval. Must be positive. Temperature checks also happen on this interval. |
| `eval_every` | `int` | Validation interval in optimizer steps. Requires either `val_glob` or `auto_val_split`. |
| `eval_batches` | `int` | Number of validation batches consumed per eval run. Use a positive value if you expect logged validation metrics. |
| `save_every` | `int` | Interval for persistent `step-XXXXXXXX.pt` checkpoints. `0` disables periodic step-checkpoint saves, but the final step checkpoint is still written at normal training exit. |
| `keep_last_n_checkpoints` | `int` | Number of persistent step checkpoints to keep. `0` means keep all. |
| `resume_from` | `str` | Checkpoint path, `auto`, or empty string. `auto` searches `last.pt`, `latest.pt`, and `step-*.pt`; empty string disables resume. |
| `output_dir` | `str` | Base output directory. Final run path is `output_dir/run_name/`. |
| `sample_every` | `int` | Online text-sampling interval during training. `0` disables it. |
| `sample_tokens` | `int` | Number of tokens generated for each online sample event. |
| `sample_prompt` | `str` | Prompt used for online sampling and also as default prompt in offline sampling if `--prompt` is omitted. |

### `logging`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `enable_csv` | `bool` | Write `metrics.csv`. |
| `enable_jsonl` | `bool` | Write `metrics.jsonl`. |
| `enable_tensorboard` | `bool` | Write TensorBoard scalars if TensorBoard is installed. |
| `enable_plot` | `bool` | Generate `metrics.png` from the CSV log. |
| `plot_every` | `int` | Plot refresh interval in train steps. |

Logged training metrics may include:

- `loss`
- `main_loss`
- `mtp_loss`
- `lr`
- `grad_norm`
- `tokens_seen`
- `tokens_per_sec`
- `ppl`
- `temperature_celsius`
- `thermal_pause_seconds`
- `total_thermal_pause_seconds`

Progress bar semantics:

- `loss`: current optimizer step loss
- `avg`: mean loss over the current logging window

### `monitoring`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `thermal_pause_enabled` | `bool` | Enables temperature-based training pause on log steps. |
| `thermal_max_celsius` | `float` | If current temperature is above this threshold at a log step, training pauses. |
| `thermal_resume_celsius` | `float` | Training resumes only after temperature drops to this value or lower. Must be `<= thermal_max_celsius`. |
| `thermal_poll_interval` | `float` | Seconds between temperature polls while training is paused. |
| `thermal_query_command` | `str` | Optional custom shell command used to read temperature. The command output must contain a numeric temperature. Useful for `XPU` or non-standard environments. |
| `status_api_enabled` | `bool` | Enables the local REST status API. |
| `status_api_host` | `str` | Bind address for the local REST status API. |
| `status_api_port` | `int` | Port for the local REST status API. |

Default automatic temperature backend:

- `CUDA`: uses `nvidia-smi --query-gpu=temperature.gpu`
- `CPU` / `XPU`: no automatic backend; use `thermal_query_command`

Example monitoring block:

```json
{
  "monitoring": {
    "thermal_pause_enabled": true,
    "thermal_max_celsius": 83.0,
    "thermal_resume_celsius": 75.0,
    "thermal_poll_interval": 15.0,
    "thermal_query_command": "",
    "status_api_enabled": true,
    "status_api_host": "127.0.0.1",
    "status_api_port": 8008
  }
}
```

## Checkpoints and Resume

Each run writes checkpoints under `outputs/<run_name>/checkpoints/`:

- `step-XXXXXXXX.pt`
  - persistent checkpoints saved on `trainer.save_every`
- `last.pt`
  - newest stable recovery point
  - refreshed on log steps and on save steps
- `latest.pt`
  - previous stable recovery point
  - retained as an additional fallback if interruption happens during `last.pt` refresh

Behavior summary:

- save steps always produce a persistent `step-XXXXXXXX.pt`
- log steps refresh `last.pt` asynchronously when possible
- `last.pt` refresh is done in the background to reduce training stalls
- `auto resume` chooses the highest valid step across:
  - `last.pt`
  - `latest.pt`
  - `step-*.pt`
- checkpoints written from compiled models can be resumed into uncompiled models, and the reverse path is also supported

Example:

```bash
python -m nanogpt_next.train ^
  --config configs/pretrain_300m_1k.json ^
  --resume-from auto
```

Or resume from a specific checkpoint:

```bash
python -m nanogpt_next.train ^
  --config configs/pretrain_300m_1k.json ^
  --resume-from outputs/pretrain-300m-1k/checkpoints/step-00010000.pt
```

## Validation Behavior

Validation runs only when both conditions are satisfied:

- `trainer.eval_every > 0`
- validation data exists, either through:
  - `data.val_glob`
  - or `data.auto_val_split = true`

If `eval_every > 0` but no validation stream is configured, training prints a warning and skips validation.

## Temperature Throttling

When `monitoring.thermal_pause_enabled=true`, training checks temperature on each `log_every` step.

Behavior:

- if temperature is at or below `thermal_max_celsius`, training continues normally
- if temperature is above `thermal_max_celsius`, training pauses
- while paused, temperature is polled every `thermal_poll_interval` seconds
- training resumes only when temperature drops to `thermal_resume_celsius` or lower
- pause duration is recorded in metrics and exposed through the status API

A custom temperature command can be used when `nvidia-smi` is not available:

```json
{
  "monitoring": {
    "thermal_pause_enabled": true,
    "thermal_query_command": "python scripts/read_xpu_temp.py"
  }
}
```

The command output only needs to contain one numeric value, for example `72` or `72.5 C`.

## REST Status API

When `monitoring.status_api_enabled=true`, training starts a local HTTP server and exposes:

- `GET /`
- `GET /status`
- `GET /healthz`

Example:

```bash
curl http://127.0.0.1:8008/status
```

Typical response fields:

- `status`
- `phase`
- `step`
- `progress`
- `max_steps`
- `tokens_seen`
- `tokens_per_step`
- `tokens_per_sec`
- `loss`
- `main_loss`
- `mtp_loss`
- `ppl`
- `lr`
- `grad_norm`
- `temperature_celsius`
- `thermal_paused`
- `thermal_pause_seconds`
- `total_thermal_pause_seconds`
- `last_checkpoint_step`
- `last_checkpoint_path`
- `val_loss`
- `val_ppl`
- `last_sample_step`
- `status_api_url`

This API is local-only by default because the default bind host is `127.0.0.1`.

## Sampling and PPL

Single-checkpoint sampling:

```bash
python -m nanogpt_next.sample ^
  --checkpoint outputs/pretrain-300m-1k/checkpoints/last.pt ^
  --prompt "The future of open language models is" ^
  --max-new-tokens 128 ^
  --temperature 0.8 ^
  --top-k 50
```

Batch sampling with wildcard checkpoints:

```bash
python -m nanogpt_next.sample ^
  --checkpoint "outputs/pretrain-300m-1k/checkpoints/step-*.pt" ^
  --prompt "The future of open language models is" ^
  --max-new-tokens 128 ^
  --output-dir outputs/pretrain-300m-1k/batch-samples
```

Optional export flags:

- `--output`
  - only for single-checkpoint mode
  - writes raw text output
- `--output-dir`
  - writes one text file per checkpoint in batch mode
- `--export-jsonl`
  - writes all sampled results to JSONL
- `--export-csv`
  - writes all sampled results to CSV

Sampling result fields:

- `checkpoint`
- `step`
- `prompt`
- `text`
- `ppl`
- `max_new_tokens`
- `temperature`
- `top_k`
- `device`
- `precision`

`PPL` is computed on the generated sequence by feeding it back through the model and averaging token-level negative log-likelihood over the generated text window.

## Outputs

Each run writes into `outputs/<run_name>/`:

- `config.resolved.json`
- `metrics.csv`
- `metrics.jsonl`
- `metrics.png`
- `tensorboard/` when TensorBoard is enabled and installed
- `checkpoints/last.pt`
- `checkpoints/latest.pt`
- `checkpoints/step-XXXXXXXX.pt`
- `samples/` when `trainer.sample_every > 0`

## Notes

- `eval_every` by itself does not create validation. You still need either `val_glob` or `auto_val_split=true`.
- `sample_every=0` only disables online sampling during training. Offline sampling from saved checkpoints still works.
- `keep_last_n_checkpoints=0` means persistent step checkpoints are not pruned.
- if `compile_model=true`, resume and sample remain compatible with checkpoints saved before or after compile.
- automatic thermal throttling does not currently auto-detect `XPU` temperatures unless you provide `thermal_query_command`.

