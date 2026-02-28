# RoadMap.md — <1B LLM Pretraining (Dense) Plan

## 0) Project Goal
Train two compact decoder-only LLMs as a solid pretrain foundation:

- **Model A:** ~**300M** params, **1K** context (fast iteration / stability baseline)
- **Model B:** ~**500M** params, **4K** context (scaled-up recipe / better utility)

Primary outcome: reproducible training + evaluation pipeline, with a clean path to longer context and larger models later.

---

## 1) Chosen Technical Path (v1)
### 1.1 Core Architecture
- Decoder-only Transformer (dense)
- **RMSNorm**
- **SwiGLU** FFN
- **RoPE** with **partial rotary** (configurable)
- **GQA** (Grouped Query Attention)
- **MTP** (Multi-Token Prediction) auxiliary head (configurable)

### 1.2 Optimization / Training
- Optimizer: **Moun + AdamW-style decoupled weight decay** (as selected)
- Scheduler: warmup → cosine decay (default), gradient clipping enabled
- Precision:
  - default compute: **BF16**
  - CUDA optional: **FP8** (only when supported by the stack; fallback to BF16)

### 1.3 Framework / Stack
- **PyTorch** training core (device-agnostic)
  - supports **CUDA** and **XPU**
- **PyTorch Lightning** for training loops, checkpointing, logging
- Tokenizer: [**TokenFluxPlusPlus**](https://github.com/TabNahida/TokenFluxPlusPlus) library integration

---

## 2) Repository Layout (target)
```

.
├── configs/
│   ├── model/
│   ├── data/
│   ├── train/
│   └── experiments/
├── src/
│   ├── models/
│   ├── layers/
│   ├── optim/
│   ├── data/
│   ├── eval/
│   └── utils/
├── scripts/
│   ├── prepare_data/
│   ├── train/
│   ├── eval/
│   └── export/
├── tests/
├── tools/
├── RoadMap.md
└── README.md

```

---

## 3) Milestones (work packages + acceptance checks)

### M0 — Bootstrap & Reproducibility
**Tasks**
- [ ] Lightning project skeleton (train/eval entrypoints)
- [ ] Unified config system (YAML + overrides; save full resolved config in runs)
- [ ] Determinism controls (seed, dataloader worker seeding)
- [ ] CI: lint + unit tests + minimal forward pass test

**Done when**
- Training script runs a dummy model for 100 steps and resumes from checkpoint identically.

---

### M1 — Tokenization (TokenFluxPlusPlus)
**Tasks**
- [ ] TokenFluxPlusPlus wrapper: encode/decode + special tokens + chat template hooks (optional)
- [ ] Export tokenizer artifacts (vocab, merges/model, config)
- [ ] Golden tests (roundtrip + expected token IDs for known strings)

**Done when**
- Tokenizer is a drop-in dependency for dataset packing + inference with stable special-token behavior.

---

### M2 — Data Pipeline (Pretrain)
**Tasks**
- [ ] Dataset format definition (document packing, sequence length handling, EOS policy)
- [ ] Streaming dataloader (shards; deterministic shuffle; multi-worker safe)
- [ ] Data quality gates (dedup hooks, language filters, code/text mixing ratios)
- [ ] Packed sequence builder for:
  - Model A: 1K context
  - Model B: 4K context

**Done when**
- You can reproduce identical batches given the same seed + shard list; throughput is stable.

---

### M3 — Model Implementation (Dense Baseline)
**Tasks**
- [ ] RMSNorm layer (fused optional)
- [ ] SwiGLU FFN (fused optional)
- [ ] RoPE module with:
  - [ ] partial rotary factor
  - [ ] configurable theta
- [ ] Attention module with **GQA**
- [ ] Full model: embeddings → blocks → lm_head
- [ ] Unit tests:
  - shape/grad checks
  - RoPE correctness sanity checks
  - GQA equivalence tests (small cases)

**Done when**
- Forward/backward works on CPU, CUDA, XPU (when available), and matches reference numerics within tolerance.

---

### M4 — Add MTP (Multi-Token Prediction)
**Tasks**
- [ ] MTP head implementation (configurable N future tokens, start with N=2)
- [ ] Loss composition: next-token + λ * MTP loss (λ configurable)
- [ ] Training/eval logging for both losses
- [ ] Ablation switch: MTP on/off

**Done when**
- You can run identical configs with MTP toggled and compare validation perplexity + training stability.

---

### M5 — Training Recipe v1: Model A (300M, 1K)
**Tasks**
- [ ] Base hyperparams (batch size, lr, warmup, wd, grad clip)
- [ ] Stable mixed precision BF16
- [ ] Checkpointing policy (steps-based + best-val)
- [ ] Early “sanity eval”: perplexity curves + loss smoothness + sample generations

**Done when**
- Model A reaches a clear downward PPL trend and can generate coherent text for basic prompts.

---

### M6 — Training Recipe v1: Model B (500M, 4K)
**Tasks**
- [ ] Scale config (layers/width/heads) + memory plan
- [ ] Sequence packing for 4K
- [ ] Throughput tuning: activation checkpointing, fused kernels where safe
- [ ] Stability tuning (grad clip, lr, wd, norm eps)

**Done when**
- Model B trains stably with comparable or better validation perplexity than Model A, and handles longer prompts reliably.

---

### M7 — Evaluation Harness (Continuous)
**Tasks**
- [ ] Perplexity on held-out corpora (general + code + bilingual)
- [ ] Lightweight downstream probes (few-shot QA, math snippets, code completion)
- [ ] Regression tests for evaluation (same checkpoint → same metrics)

**Done when**
- Each experiment produces a single comparable metric bundle + generation samples artifact.

---

### M8 — Performance & Precision Upgrades
**Tasks**
- [ ] CUDA: optional FP8 path (guarded behind capability checks; auto-fallback to BF16)
- [ ] Attention speedups (FlashAttention if compatible; otherwise baseline)
- [ ] Compile options (torch.compile / SDPA selection)
- [ ] XPU performance pass (ensure no CUDA-only assumptions)

**Done when**
- Same model config runs across devices with consistent results; speedups do not break reproducibility.

---

### M9 — Packaging & Release
**Tasks**
- [ ] Export to a standard checkpoint format + tokenizer bundle
- [ ] Minimal inference script (greedy + sampling)
- [ ] Model card template: data summary, architecture, known limits, eval metrics

**Done when**
- A clean “train → export → run inference” story exists for both Model A and B.

---

## 4) Experiment Matrix (minimum ablations)
Run these on Model A first (cheap), then confirm on Model B (important).

### Architecture ablations
- [ ] GQA: (num_kv_heads = 1/2/4) vs baseline
- [ ] partial rotary factor: 0.25 vs 0.5 vs 1.0
- [ ] MTP: off vs on (N=2; λ sweep)

### Optimization ablations
- [ ] Moun+AdamW baseline vs (pure AdamW if needed for fallback)
- [ ] WD sweep (e.g., 0.05 / 0.1 / 0.2)
- [ ] LR sweep (small grid)
- [ ] grad clip thresholds

### Context scaling checks
- [ ] 1K-trained model tested on 2K (stress)
- [ ] 4K-trained model tested on 8K (stress; expect degradation but no collapse)

---

## 5) Configuration Targets (initial defaults)
These are placeholders; keep them in `configs/` and tune via sweeps.

- Precision: BF16 default; CUDA FP8 optional
- Scheduler: warmup_ratio ~ 0.01–0.03 then cosine
- Gradient clipping: enabled (start modest, tune)
- Logging: steps-based (loss, lr, grad norm, tokens/sec, memory)

---

## 6) Risk List (and mitigations)
- **Training instability at 4K** → tighten grad clip, lower LR, raise norm eps, check RoPE + attention numerics
- **Tokenizer drift / special token mismatch** → lock tokenizer version + golden tests
- **Device divergence (XPU vs CUDA)** → keep numerically stable ops, avoid CUDA-only kernels without guards
- **FP8 causing silent regressions** → only enable with explicit flag + metric parity check vs BF16

---

## 7) Definition of “Success” (v1)
- Both models train end-to-end reproducibly with:
  - stable loss curves
  - comparable evaluation bundles
  - exportable checkpoints + runnable inference
- Clear ablation evidence for:
  - MTP benefit (or not)
  - partial rotary choice
  - GQA settings
- Pipeline is ready to extend to longer context and/or hybrid attention in v2.

