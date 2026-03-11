from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .checkpoint import (
    AsyncLastCheckpointSaver,
    clone_to_cpu,
    find_latest_checkpoint,
    load_checkpoint,
    promote_checkpoint_to_last,
    save_checkpoint,
)
from .config import ExperimentConfig, load_config, save_config
from .data import build_tokenizer_and_dataloaders
from .model import GPTModel
from .optim import OptimizerBundle, build_optimizers
from .tokenizer import TokenizerAdapter
from .train_logging import MetricsLogger
from .utils import autocast_kwargs, cosine_lr_multiplier, detect_device, format_num, load_env_file, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NanoGPT-Next models.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment JSON config.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--train-glob", type=str, default=None)
    parser.add_argument("--val-glob", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    return parser.parse_args()


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.max_steps is not None:
        config.trainer.max_steps = args.max_steps
        config.trainer.lr_decay_steps = max(config.trainer.lr_decay_steps, args.max_steps)
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.output_dir is not None:
        config.trainer.output_dir = str(args.output_dir)
    if args.resume_from is not None:
        config.trainer.resume_from = args.resume_from
    if args.train_glob is not None:
        config.data.train_glob = args.train_glob
    if args.val_glob is not None:
        config.data.val_glob = args.val_glob
    if args.device is not None:
        config.trainer.device = args.device
    if args.precision is not None:
        config.trainer.precision = args.precision
    config.validate()
    return config


def prepare_run_dir(config: ExperimentConfig) -> Path:
    run_dir = Path(config.trainer.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def maybe_enable_float8(model: GPTModel, config: ExperimentConfig, device: torch.device) -> None:
    if config.trainer.precision != "fp8":
        return
    if device.type != "cuda":
        raise ValueError("FP8 mode requires a CUDA device.")
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    convert_to_float8_training(model, config=Float8LinearConfig())


def maybe_compile_model(model: GPTModel, enabled: bool) -> GPTModel:
    if not enabled:
        return model
    return torch.compile(model)  # type: ignore[return-value]


def maybe_resume(
    model: GPTModel,
    optimizer_bundle: OptimizerBundle,
    scaler: torch.cuda.amp.GradScaler | None,
    run_dir: Path,
    resume_from: str,
) -> tuple[int, int]:
    ckpt_path: Path | None = None
    if resume_from == "auto":
        ckpt_path = find_latest_checkpoint(run_dir)
    elif resume_from:
        ckpt_path = Path(resume_from)

    if ckpt_path is None or not ckpt_path.exists():
        return 0, 0

    checkpoint = load_checkpoint(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    optimizer_bundle.load_state_dict(checkpoint["optimizers"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("step", 0)), int(checkpoint.get("tokens_seen", 0))


def build_checkpoint_payload(
    config: ExperimentConfig,
    model: GPTModel,
    optimizer_bundle: OptimizerBundle,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    tokens_seen: int,
) -> dict[str, object]:
    return {
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizers": optimizer_bundle.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
    }


@torch.no_grad()
def evaluate(
    model: GPTModel,
    data_iter,
    steps: int,
    device: torch.device,
    autocast_context,
) -> dict[str, float]:
    losses = []
    main_losses = []
    mtp_losses = []
    model.eval()
    iterator = iter(data_iter)
    for _ in range(steps):
        try:
            input_ids, targets = next(iterator)
        except StopIteration:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context():
            output = model(input_ids, targets)
        if output.loss is None or output.main_loss is None:
            continue
        losses.append(float(output.loss.detach().cpu()))
        main_losses.append(float(output.main_loss.detach().cpu()))
        if output.mtp_losses:
            mtp_losses.append(float(torch.stack(output.mtp_losses).mean().detach().cpu()))
    model.train()
    if not losses:
        return {}
    return {
        "loss": sum(losses) / len(losses),
        "main_loss": sum(main_losses) / len(main_losses),
        "mtp_loss": sum(mtp_losses) / len(mtp_losses) if mtp_losses else 0.0,
    }


def sample_text(
    model: GPTModel,
    tokenizer: TokenizerAdapter,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    if not prompt:
        return ""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        return ""
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)[0].tolist()
    return tokenizer.decode(output_ids)


def main() -> None:
    load_env_file()
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    seed_everything(config.seed)
    torch.set_float32_matmul_precision(config.trainer.matmul_precision)
    device = detect_device(config.trainer.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = prepare_run_dir(config)
    save_config(config, run_dir / "config.resolved.json")

    tokenizer, train_loader, val_loader = build_tokenizer_and_dataloaders(
        config.data,
        seq_len=config.model.max_seq_len,
        seed=config.seed,
        micro_batch_size=config.trainer.micro_batch_size,
    )
    if config.data.auto_val_split and not config.data.val_glob:
        print(
            "info: validation data is auto-split from train_glob "
            f"(ratio={config.data.auto_val_ratio:.4f}, seed={config.data.auto_val_seed})."
        )
    if config.trainer.eval_every > 0 and val_loader is None:
        print("warning: eval_every > 0 but no validation dataset is configured; set data.val_glob or pass --val-glob.")
    if config.model.vocab_size != tokenizer.vocab_size:
        config.model.vocab_size = tokenizer.vocab_size
        save_config(config, run_dir / "config.resolved.json")

    model = GPTModel(config.model)
    maybe_enable_float8(model, config, device)
    model = model.to(device)
    optimizer_bundle = build_optimizers(model, config.optimizer, device)

    scaler: torch.cuda.amp.GradScaler | None = None
    if config.trainer.precision == "fp16" and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_step, tokens_seen = maybe_resume(
        model,
        optimizer_bundle,
        scaler,
        run_dir,
        config.trainer.resume_from,
    )

    model = maybe_compile_model(model, config.trainer.compile_model)

    enable_autocast, autocast_dtype = autocast_kwargs(device, config.trainer.precision)
    if config.trainer.precision == "fp8":
        enable_autocast = device.type == "cuda"
        autocast_dtype = torch.bfloat16

    def autocast_context():
        if enable_autocast and autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    logger = MetricsLogger(run_dir, config.logging)
    last_checkpoint_saver = AsyncLastCheckpointSaver(run_dir)

    total_params = model.num_parameters() if hasattr(model, "num_parameters") else sum(
        parameter.numel() for parameter in model.parameters()
    )
    tokens_per_step = (
        config.trainer.micro_batch_size
        * config.trainer.gradient_accumulation_steps
        * config.model.max_seq_len
    )
    print(
        f"run_dir={run_dir} device={device} precision={config.trainer.precision} "
        f"params={format_num(total_params)} tokens_per_step={format_num(tokens_per_step)}"
    )

    progress = tqdm(
        range(start_step, config.trainer.max_steps),
        initial=start_step,
        total=config.trainer.max_steps,
        dynamic_ncols=True,
    )
    train_iter = iter(train_loader)
    optimizer_bundle.zero_grad(set_to_none=True)

    window_loss = 0.0
    window_main_loss = 0.0
    window_mtp_loss = 0.0
    window_steps = 0
    last_log_time = time.perf_counter()
    last_saved_step = start_step

    try:
        for step in progress:
            lr_mult = cosine_lr_multiplier(
                step=step,
                warmup_steps=config.trainer.warmup_steps,
                decay_steps=config.trainer.lr_decay_steps,
                min_lr_ratio=config.trainer.min_lr_ratio,
            )
            optimizer_bundle.set_lrs(
                adamw_lr=config.optimizer.adamw_lr * lr_mult,
                muon_lr=config.optimizer.muon_lr * lr_mult,
            )

            step_loss_total = 0.0
            step_main_loss_total = 0.0
            step_mtp_loss_total = 0.0

            for micro_step in range(config.trainer.gradient_accumulation_steps):
                input_ids, targets = next(train_iter)
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast_context():
                    output = model(input_ids, targets)
                    loss = output.loss
                    if loss is None or output.main_loss is None:
                        raise RuntimeError("Model did not produce training loss.")
                    scaled_loss = loss / config.trainer.gradient_accumulation_steps
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                step_loss_total += float(loss.detach().cpu())
                step_main_loss_total += float(output.main_loss.detach().cpu())
                if output.mtp_losses:
                    step_mtp_loss_total += float(torch.stack(output.mtp_losses).mean().detach().cpu())
                if micro_step + 1 == config.trainer.gradient_accumulation_steps:
                    break

            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer_bundle.adamw)
                if optimizer_bundle.muon is not None:
                    scaler.unscale_(optimizer_bundle.muon)

            grad_norm = float(clip_grad_norm_(model.parameters(), config.trainer.clip_grad_norm))
            optimizer_bundle.step(scaler)
            optimizer_bundle.zero_grad(set_to_none=True)

            tokens_seen += tokens_per_step
            current_step = step + 1
            step_loss = step_loss_total / config.trainer.gradient_accumulation_steps
            step_main_loss = step_main_loss_total / config.trainer.gradient_accumulation_steps
            step_mtp_loss = step_mtp_loss_total / config.trainer.gradient_accumulation_steps
            should_save_step = config.trainer.save_every > 0 and current_step % config.trainer.save_every == 0

            window_loss += step_loss
            window_main_loss += step_main_loss
            window_mtp_loss += step_mtp_loss
            window_steps += 1

            progress.set_postfix(
                loss=f"{step_loss:.4f}",
                avg=f"{window_loss / max(window_steps, 1):.4f}",
                lr=f"{config.optimizer.adamw_lr * lr_mult:.2e}",
                grad=f"{grad_norm:.2f}",
            )

            if current_step % config.trainer.log_every == 0 or current_step == 1:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-6)
                tokens_per_sec = (tokens_per_step * window_steps) / elapsed
                avg_loss = window_loss / max(window_steps, 1)
                avg_main_loss = window_main_loss / max(window_steps, 1)
                avg_mtp_loss = window_mtp_loss / max(window_steps, 1)
                metrics = {
                    "loss": avg_loss,
                    "main_loss": avg_main_loss,
                    "mtp_loss": avg_mtp_loss,
                    "lr": config.optimizer.adamw_lr * lr_mult,
                    "grad_norm": grad_norm,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tokens_per_sec,
                    "ppl": math.exp(min(avg_main_loss, 20.0)),
                }
                logger.log(current_step, "train", metrics)
                if not should_save_step and not last_checkpoint_saver.has_pending_work():
                    payload = build_checkpoint_payload(
                        config=config,
                        model=model,
                        optimizer_bundle=optimizer_bundle,
                        scaler=scaler,
                        step=current_step,
                        tokens_seen=tokens_seen,
                    )
                    last_checkpoint_saver.submit(clone_to_cpu(payload))
                window_loss = 0.0
                window_main_loss = 0.0
                window_mtp_loss = 0.0
                window_steps = 0
                last_log_time = now

            if val_loader is not None and config.trainer.eval_every > 0 and current_step % config.trainer.eval_every == 0:
                val_metrics = evaluate(
                    model=model,
                    data_iter=val_loader,
                    steps=config.trainer.eval_batches,
                    device=device,
                    autocast_context=autocast_context,
                )
                if val_metrics:
                    val_metrics["ppl"] = math.exp(min(val_metrics["main_loss"], 20.0))
                    logger.log(current_step, "val", val_metrics)

            if config.trainer.sample_every > 0 and current_step % config.trainer.sample_every == 0:
                sample = sample_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=config.trainer.sample_prompt,
                    max_new_tokens=config.trainer.sample_tokens,
                    device=device,
                )
                if sample:
                    logger.log_text("sample", sample, current_step)

            if should_save_step:
                last_checkpoint_saver.flush()
                ckpt_payload = build_checkpoint_payload(
                    config=config,
                    model=model,
                    optimizer_bundle=optimizer_bundle,
                    scaler=scaler,
                    step=current_step,
                    tokens_seen=tokens_seen,
                )
                ckpt_path = save_checkpoint(
                    run_dir=run_dir,
                    step=current_step,
                    payload=ckpt_payload,
                    keep_last_n=config.trainer.keep_last_n_checkpoints,
                )
                promote_checkpoint_to_last(run_dir=run_dir, checkpoint_path=ckpt_path)
                last_saved_step = current_step

        if last_saved_step != config.trainer.max_steps:
            last_checkpoint_saver.flush()
            ckpt_payload = build_checkpoint_payload(
                config=config,
                model=model,
                optimizer_bundle=optimizer_bundle,
                scaler=scaler,
                step=config.trainer.max_steps,
                tokens_seen=tokens_seen,
            )
            ckpt_path = save_checkpoint(
                run_dir=run_dir,
                step=config.trainer.max_steps,
                payload=ckpt_payload,
                keep_last_n=config.trainer.keep_last_n_checkpoints,
            )
            promote_checkpoint_to_last(run_dir=run_dir, checkpoint_path=ckpt_path)
    finally:
        last_checkpoint_saver.close()
        logger.close()


if __name__ == "__main__":
    main()
