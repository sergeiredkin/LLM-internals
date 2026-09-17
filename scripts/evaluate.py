#!/usr/bin/env python3
"""Deterministically evaluate a checkpoint and write a reproducible JSON result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from llm.config import config_from_dict
from llm.data import TokenCorpus
from llm.model import GPT
from llm.training import (
    autocast_context,
    estimate_loss,
    evaluation_generator,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-seed", type=int, default=2025)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--prompt", default="Once upon a time, there was a little fox named Pip.")
    parser.add_argument("--generation-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.batches <= 0 or args.batch_size <= 0:
        raise SystemExit("--batches and --batch-size must be positive")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise SystemExit("Evaluation expects a full format-version-1 checkpoint")
    config = config_from_dict(checkpoint["config"])
    corpus = TokenCorpus(config.data.processed_dir)
    model = GPT(config.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    validation_loss = estimate_loss(
        model,
        corpus,
        "val",
        args.batches,
        args.batch_size,
        config.model.context_length,
        device,
        config.training.precision,
        evaluation_generator(args.evaluation_seed, "val"),
    )
    synchronize(device)
    evaluation_seconds = time.perf_counter() - started
    evaluated_tokens = args.batches * args.batch_size * config.model.context_length

    prompt_ids = corpus.tokenizer.encode(args.prompt)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    torch.manual_seed(args.generation_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.generation_seed)
    synchronize(device)
    generation_started = time.perf_counter()
    with autocast_context(device, config.training.precision):
        generated = model.generate(
            prompt,
            max_new_tokens=args.generation_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=getattr(corpus.tokenizer, "eos_id", None),
        )
    synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    generated_tokens = generated.shape[1] - prompt.shape[1]
    sample = corpus.tokenizer.decode(generated[0])

    peak_vram = (
        torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
    )
    result = {
        "evaluation_version": 1,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "training_best_val_loss": float(checkpoint["best_val_loss"]),
        "model": {
            "parameters": model.num_parameters(),
            "vocab_size": config.model.vocab_size,
            "context_length": config.model.context_length,
            "n_layers": config.model.n_layers,
            "d_model": config.model.d_model,
            "n_heads": config.model.n_heads,
            "n_kv_heads": config.model.n_kv_heads,
            "position_encoding": config.model.position_encoding,
            "mlp_type": config.model.mlp_type,
        },
        "validation": {
            "seed": args.evaluation_seed,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "tokens": evaluated_tokens,
            "loss": validation_loss,
            "perplexity": math.exp(validation_loss),
            "seconds": evaluation_seconds,
            "tokens_per_second": evaluated_tokens / evaluation_seconds,
        },
        "generation": {
            "seed": args.generation_seed,
            "prompt": args.prompt,
            "max_new_tokens": args.generation_tokens,
            "generated_tokens": generated_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seconds": generation_seconds,
            "tokens_per_second": generated_tokens / max(generation_seconds, 1e-9),
            "sample": sample,
        },
        "peak_inference_vram_gib": peak_vram,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Checkpoint step:     {result['checkpoint_step']:,}")
    print(f"Parameters:          {result['model']['parameters']:,}")
    print(f"Validation loss:     {validation_loss:.4f}")
    print(f"Perplexity:          {math.exp(validation_loss):.2f}")
    print(f"Evaluation speed:    {evaluated_tokens / evaluation_seconds:,.0f} tok/s")
    print(f"Generation speed:    {result['generation']['tokens_per_second']:.1f} tok/s")
    print(f"Peak inference VRAM: {peak_vram:.2f} GiB")
    print(f"Result:              {args.output}")
    print("\nSample\n------")
    print(sample)


if __name__ == "__main__":
    main()
