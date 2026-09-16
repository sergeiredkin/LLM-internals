#!/usr/bin/env python3
"""Export a compact, portable, inference-only checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llm.config import config_from_dict
from llm.model import GPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tinystories/processed/tokenizer.json"),
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if source.get("format_version") != 1:
        raise SystemExit("Export expects a full format-version-1 checkpoint")
    config = config_from_dict(source["config"])
    model = GPT(config.model)
    model.load_state_dict(source["model"])
    if args.dtype == "bf16":
        model.to(dtype=torch.bfloat16)

    bundle = {
        "format_version": 2,
        "kind": "inference",
        "dtype": args.dtype,
        "source_step": int(source["step"]),
        "source_best_val_loss": float(source["best_val_loss"]),
        "config": config.to_dict(),
        "tokenizer_json": args.tokenizer.read_text(encoding="utf-8"),
        "model": model.state_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(args.output)
    checksum = sha256(args.output)
    metadata = {
        "file": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": checksum,
        "dtype": args.dtype,
        "parameters": model.num_parameters(),
        "source_checkpoint": str(args.checkpoint),
        "source_step": source["step"],
        "validation_loss": source["best_val_loss"],
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported:   {args.output}")
    print(f"Size:       {args.output.stat().st_size / 2**20:.1f} MiB")
    print(f"SHA-256:    {checksum}")
    print("Omitted:    optimizer, scaler, and RNG state")


if __name__ == "__main__":
    main()
