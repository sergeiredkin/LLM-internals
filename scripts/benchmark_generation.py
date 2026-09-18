#!/usr/bin/env python3
"""Benchmark cached and uncached greedy decoding with identical tokens."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from llm.config import config_from_dict
from llm.data import TokenCorpus
from llm.model import GPT
from llm.training import autocast_context, resolve_device


def parse_lengths(value: str) -> list[int]:
    try:
        lengths = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers") from error
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("all lengths must be positive")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("64,128,256,400"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def greedy_decode(
    model: GPT,
    prompt: torch.Tensor,
    new_tokens: int,
    use_cache: bool,
    cache_dtype: torch.dtype,
) -> tuple[torch.Tensor, float, float, float]:
    """Return tokens, prefill seconds, decode seconds, and peak allocated GiB."""

    device = prompt.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    caches = (
        model.create_kv_caches(
            prompt.shape[0],
            max_length=prompt.shape[1] + new_tokens,
            device=device,
            dtype=cache_dtype,
        )
        if use_cache
        else None
    )
    logits, _ = model(prompt, caches=caches)
    synchronize(device)
    prefill_seconds = time.perf_counter() - started

    generated = prompt
    synchronize(device)
    started = time.perf_counter()
    for step in range(new_tokens):
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        if step + 1 == new_tokens:
            break
        if use_cache:
            logits, _ = model(next_token, caches=caches)
        else:
            logits, _ = model(generated)
    synchronize(device)
    decode_seconds = time.perf_counter() - started
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    )
    return generated, prefill_seconds, decode_seconds, peak_gib


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.warmup_tokens <= 0:
        raise SystemExit("--repeats and --warmup-tokens must be positive")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_dict(checkpoint["config"])
    corpus = TokenCorpus(config.data.processed_dir)
    prompt_ids = corpus.tokenizer.encode(args.prompt)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if prompt.shape[1] + max(args.lengths) > config.model.context_length:
        raise SystemExit("prompt plus largest generation length exceeds model context")

    model = GPT(config.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    cache_dtype = (
        torch.bfloat16
        if config.training.precision == "bf16" and device.type == "cuda"
        else model.token_embedding.weight.dtype
    )

    records: list[dict[str, object]] = []
    with torch.inference_mode(), autocast_context(device, config.training.precision):
        # Warm both paths before collecting measurements.
        greedy_decode(model, prompt, args.warmup_tokens, False, cache_dtype)
        greedy_decode(model, prompt, args.warmup_tokens, True, cache_dtype)

        for length in args.lengths:
            timings: dict[str, list[dict[str, float]]] = {"uncached": [], "cached": []}
            reference: torch.Tensor | None = None
            exact_match = True
            for _ in range(args.repeats):
                for label, use_cache in (("uncached", False), ("cached", True)):
                    tokens, prefill, decode, peak = greedy_decode(
                        model, prompt, length, use_cache, cache_dtype
                    )
                    if reference is None:
                        reference = tokens
                    else:
                        exact_match = exact_match and torch.equal(tokens, reference)
                    timings[label].append(
                        {
                            "prefill_seconds": prefill,
                            "decode_seconds": decode,
                            "total_seconds": prefill + decode,
                            "peak_allocated_gib": peak,
                        }
                    )

            summary: dict[str, object] = {
                "new_tokens": length,
                "exact_token_match": exact_match,
            }
            for label in ("uncached", "cached"):
                values = timings[label]
                prefill = statistics.median(item["prefill_seconds"] for item in values)
                decode = statistics.median(item["decode_seconds"] for item in values)
                total = statistics.median(item["total_seconds"] for item in values)
                summary[label] = {
                    "prefill_seconds": prefill,
                    "decode_seconds": decode,
                    "total_seconds": total,
                    "decode_tokens_per_second": length / decode,
                    "total_tokens_per_second": length / total,
                    "peak_allocated_gib": max(item["peak_allocated_gib"] for item in values),
                    "runs": values,
                }
            cached_total = summary["cached"]["total_seconds"]  # type: ignore[index]
            uncached_total = summary["uncached"]["total_seconds"]  # type: ignore[index]
            summary["total_speedup"] = uncached_total / cached_total
            records.append(summary)
            print(
                f"{length:4d} tokens | match={exact_match} | "
                f"uncached={uncached_total:.3f}s | cached={cached_total:.3f}s | "
                f"speedup={uncached_total / cached_total:.2f}x"
            )

    cache_bytes = (
        config.model.n_layers
        * 2
        * prompt.shape[0]
        * config.model.n_kv_heads
        * config.model.context_length
        * config.model.head_dim
        * torch.tensor([], dtype=cache_dtype).element_size()
    )
    result = {
        "benchmark_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step", checkpoint.get("source_step")),
        "device": str(device),
        "precision": config.training.precision,
        "prompt": args.prompt,
        "prompt_tokens": prompt.shape[1],
        "repeats": args.repeats,
        "model": {
            "parameters": model.num_parameters(),
            "context_length": config.model.context_length,
            "n_layers": config.model.n_layers,
            "n_heads": config.model.n_heads,
            "n_kv_heads": config.model.n_kv_heads,
            "head_dim": config.model.head_dim,
        },
        "full_capacity_cache_bytes": cache_bytes,
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
