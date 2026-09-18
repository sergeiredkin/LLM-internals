#!/usr/bin/env python3
"""Generate text from a learnGPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm.config import config_from_dict
from llm.data import TokenCorpus
from llm.model import GPT
from llm.tokenizer import tokenizer_from_json
from llm.training import autocast_context, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="prefill once and reuse per-layer keys and values during decoding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens cannot be negative")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_format = checkpoint.get("format_version")
    if checkpoint_format not in {1, 2}:
        raise SystemExit("Unsupported checkpoint format")
    config = config_from_dict(checkpoint["config"])
    if checkpoint_format == 2:
        tokenizer = tokenizer_from_json(checkpoint["tokenizer_json"])
        checkpoint_step = checkpoint["source_step"]
    else:
        tokenizer = TokenCorpus(config.data.processed_dir).tokenizer
        checkpoint_step = checkpoint["step"]
    if config.model.vocab_size != tokenizer.vocab_size:
        raise SystemExit("Checkpoint and tokenizer vocabulary sizes do not match")

    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except ValueError as error:
        raise SystemExit(f"Prompt cannot be encoded: {error}") from error
    if not prompt_ids:
        raise SystemExit("Prompt cannot be empty")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = GPT(config.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with autocast_context(device, config.training.precision):
        generated = model.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=getattr(tokenizer, "eos_id", None),
            use_kv_cache=args.use_kv_cache,
        )
    text = tokenizer.decode(generated[0])

    print(
        f"checkpoint={args.checkpoint} step={checkpoint_step} "
        f"device={device} temperature={args.temperature} top_k={args.top_k} "
        f"kv_cache={args.use_kv_cache}"
    )
    print("-" * 72)
    print(text)


if __name__ == "__main__":
    main()
