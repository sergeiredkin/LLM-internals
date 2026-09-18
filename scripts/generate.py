#!/usr/bin/env python3
"""Generate text from a learnGPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm.config import config_from_dict
from llm.data import TokenCorpus
from llm.model import GPT
from llm.quantization import replace_linear_with_int4, replace_linear_with_int8
from llm.tokenizer import tokenizer_from_json
from llm.training import autocast_context, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--top-k", type=int, default=20, help="0 disables top-k filtering"
    )
    parser.add_argument(
        "--top-p", type=float, default=None, help="nucleus threshold in (0, 1]"
    )
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
    if args.top_k < 0:
        raise SystemExit("--top-k cannot be negative")
    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        raise SystemExit("--top-p must be in (0, 1]")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_format = checkpoint.get("format_version")
    if checkpoint_format not in {1, 2, 3, 4}:
        raise SystemExit("Unsupported checkpoint format")
    config = config_from_dict(checkpoint["config"])
    if checkpoint_format in {2, 3, 4}:
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
    if checkpoint.get("dtype") == "bf16":
        model.to(dtype=torch.bfloat16)
    if checkpoint_format == 3:
        if checkpoint.get("kind") != "int8-inference":
            raise SystemExit("Unsupported format-version-3 checkpoint kind")
        replace_linear_with_int8(model, exclude={"lm_head"})
    elif checkpoint_format == 4:
        if checkpoint.get("kind") != "int4-inference":
            raise SystemExit("Unsupported format-version-4 checkpoint kind")
        group_size = int(checkpoint["quantization"]["group_size"])
        replace_linear_with_int4(
            model, group_size=group_size, exclude={"lm_head"}
        )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with autocast_context(device, config.training.precision):
        generated = model.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            top_p=args.top_p,
            eos_token_id=getattr(tokenizer, "eos_id", None),
            use_kv_cache=args.use_kv_cache,
        )
    text = tokenizer.decode(generated[0])

    print(
        f"checkpoint={args.checkpoint} step={checkpoint_step} "
        f"device={device} temperature={args.temperature} top_k={args.top_k or None} "
        f"top_p={args.top_p} kv_cache={args.use_kv_cache}"
    )
    print("-" * 72)
    print(text)


if __name__ == "__main__":
    main()
