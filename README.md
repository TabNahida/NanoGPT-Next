# NanoGPT-Next

A minimal PyTorch training stack for experimenting with next-gen sub-1B language models.

## Implemented baseline

- Architecture: `RMSNorm + SwiGLU + RoPE (partial rotary) + GQA`
- Training objective: autoregressive next-token loss with lightweight `MTP` auxiliary heads
- Optimizers: `Muon` for 2D hidden-layer weights plus `AdamW` for embeddings, norms, and scalar parameters
- Devices: `CUDA`, `XPU`, and CPU fallback
- Precision:
  - default mixed precision `BF16`
  - optional experimental `FP8` training on CUDA through `torchao`
- Data:
  - parquet text streaming through `pyarrow`
  - tokenizer backend prefers `TokenFlux` when installed, otherwise falls back to the bundled `Tokenizer/V1/tokenizer.json`
- Training tooling:
  - resumable checkpoints
  - CSV / JSONL metrics logs
  - optional TensorBoard logs if `tensorboard` is installed
  - auto-generated `metrics.png`

## Provided configs

- `configs/pretrain_300m_1k.json`
  - target: ~300M class
  - context: `1024`
- `configs/pretrain_500m_4k.json`
  - target: ~500M class
  - context: `4096`
- `configs/smoke_tiny.json`
  - tiny smoke-test config

## Install

```bash
python -m pip install -e .
```

Optional visualization support:

```bash
python -m pip install -e .[viz]
```

## Data

The training loader expects parquet files with one text column such as `text`, `plain_text`, `content`, or `document`.

You can point training data to either:

- `data.train_glob` inside the JSON config
- `DATA_PATH` in `.env`

Example `.env`:

```env
DATA_PATH=/OpenWebText/plain_text/train-*-of-00080.parquet
```

## Train

```bash
python -m nanogpt_next.train --config configs/pretrain_300m_1k.json
```

Common overrides:

```bash
python -m nanogpt_next.train ^
  --config configs/pretrain_500m_4k.json ^
  --precision fp8 ^
  --device cuda ^
  --max-steps 2000
```

## Outputs

Each run writes into `outputs/<run_name>/`:

- `config.resolved.json`
- `metrics.csv`
- `metrics.jsonl`
- `metrics.png`
- `checkpoints/last.pt`
- `checkpoints/latest.pt`
- `checkpoints/step-XXXXXXXX.pt`
- `samples/` when text sampling is enabled

## Plot Metrics

```bash
python scripts/plot_metrics.py outputs/pretrain-300m-1k/metrics.csv --output outputs/pretrain-300m-1k/metrics.png
```
