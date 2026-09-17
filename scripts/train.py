#!/usr/bin/env python3
"""Train the configured decoder-only GPT model."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from llm.checkpoint import load_checkpoint, save_checkpoint
from llm.config import load_config
from llm.data import TokenCorpus
from llm.model import GPT
from llm.training import (
    create_grad_scaler,
    create_optimizer,
    estimate_loss,
    evaluation_generator,
    learning_rate_at,
    resolve_device,
    set_learning_rate,
    train_micro_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/shakespeare.yaml"))
    parser.add_argument("--resume", type=Path, help="checkpoint to resume")
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="allow training even when less than 8 GiB of VRAM is free",
    )
    return parser.parse_args()


def require_training_vram(device: torch.device, allow_shared: bool) -> None:
    if device.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info(device)
    free_gib, total_gib = free / 2**30, total / 2**30
    print(f"GPU memory: {free_gib:.2f}/{total_gib:.2f} GiB free")
    if free_gib < 8 and not allow_shared:
        raise SystemExit(
            "Less than 8 GiB VRAM is free. Stop Ollama temporarily, or pass "
            "--allow-shared-gpu to accept slower training/OOM risk."
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config.training.device)
    require_training_vram(device, args.allow_shared_gpu)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(config.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.training.seed)

    corpus = TokenCorpus(config.data.processed_dir)
    model_config = replace(config.model, vocab_size=corpus.tokenizer.vocab_size)
    config = replace(config, model=model_config)
    model = GPT(model_config).to(device)
    optimizer = create_optimizer(model, config.training)
    scaler = create_grad_scaler(device, config.training.precision)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_step, best_val_loss = 0, math.inf
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, scaler, map_location=device
        )
        start_step = int(checkpoint["step"])
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"Resumed {args.resume} at step {start_step}")

    print(f"Experiment:    {config.name}")
    print(f"Device:        {device}")
    print(f"Precision:     {config.training.precision}")
    print(f"Parameters:    {model.num_parameters():,}")
    print(f"Tokens/update: {config.tokens_per_update:,}")

    training_started = time.perf_counter()
    last_time = training_started
    for step in range(start_step, config.training.max_steps):
        if step % config.training.eval_interval == 0:
            train_loss = estimate_loss(
                model,
                corpus,
                "train",
                config.training.eval_batches,
                config.training.micro_batch_size,
                config.model.context_length,
                device,
                config.training.precision,
                evaluation_generator(config.training.seed, "train"),
            )
            val_loss = estimate_loss(
                model,
                corpus,
                "val",
                config.training.eval_batches,
                config.training.micro_batch_size,
                config.model.context_length,
                device,
                config.training.precision,
                evaluation_generator(config.training.seed, "val"),
            )
            print(
                f"eval step {step:5d} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    config,
                    step,
                    best_val_loss,
                )

        learning_rate = learning_rate_at(step, config.training)
        set_learning_rate(optimizer, learning_rate)
        loss, grad_norm = train_micro_batches(
            model,
            corpus,
            optimizer,
            scaler,
            config.training,
            config.model.context_length,
            device,
        )

        if step % config.training.log_interval == 0:
            now = time.perf_counter()
            elapsed = now - last_time
            tokens_per_second = (
                config.tokens_per_update * config.training.log_interval / elapsed
                if step > start_step
                else config.tokens_per_update / elapsed
            )
            print(
                f"step {step:5d} | loss {loss:.4f} | lr {learning_rate:.2e} | "
                f"grad {grad_norm:.3f} | {tokens_per_second:,.0f} tok/s"
            )
            last_time = now

        completed_step = step + 1
        if completed_step % config.training.checkpoint_interval == 0:
            save_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                scaler,
                config,
                completed_step,
                best_val_loss,
            )
        if config.training.step_idle_seconds:
            time.sleep(config.training.step_idle_seconds)

    save_checkpoint(
        output_dir / "latest.pt",
        model,
        optimizer,
        scaler,
        config,
        config.training.max_steps,
        best_val_loss,
    )
    training_seconds = time.perf_counter() - training_started
    hours, remainder = divmod(int(training_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training complete. Checkpoints: {output_dir}")
    print(
        f"Training wall time: {hours:02d}:{minutes:02d}:{seconds:02d} "
        f"({training_seconds:.1f} seconds)"
    )
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
        print(
            f"Peak PyTorch VRAM: {peak_allocated:.2f} GiB allocated, "
            f"{peak_reserved:.2f} GiB reserved"
        )


if __name__ == "__main__":
    main()
