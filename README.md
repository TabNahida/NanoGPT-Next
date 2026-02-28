# NanoGPT-Next

Dense decoder-only LLM pretraining baseline aligned to `RoadMap.md`.

## Implemented scope
- M0 bootstrap: project layout, YAML config system, deterministic seed utilities, training/eval entrypoints, CI
- M1 tokenizer integration: local tokenizer bundle loader (`Tokenizer/V1`), encode/decode wrapper, special-token handling, tests
- M2 data pipeline: parquet streaming, deterministic shard shuffle, sequence packing for 1K/4K
- M3 model core: RMSNorm, SwiGLU, RoPE (partial rotary), GQA attention, decoder stack
- M4 MTP: configurable multi-token prediction auxiliary head + weighted loss

## Quick start
1. Install dependencies:
```bash
python -m pip install -e .[dev]
```

2. Ensure `.env` contains `DATA_PATH`, e.g.:
```env
DATA_PATH=D:\\Data\\AI\\LLM\\Text\\OpenWebText\\plain_text\\train-*-of-00080.parquet
```

3. Run a short smoke training:
```bash
python scripts/train/train.py --config configs/experiments/model_a_1k.yaml --override train.max_steps=20 --override train.devices=1
```

4. Run tests:
```bash
pytest
```

## Config layout
- `configs/model/*.yaml`: model shapes (300M/500M)
- `configs/data/*.yaml`: data source + packing settings
- `configs/train/*.yaml`: optimizer/scheduler/runtime
- `configs/experiments/*.yaml`: composed experiment entries

## Notes
- The tokenizer wrapper tries TokenFluxPlusPlus first if installed, otherwise falls back to `tokenizers` with `Tokenizer/V1/tokenizer.json`.
- Model B (4K) enables activation checkpointing by default.
