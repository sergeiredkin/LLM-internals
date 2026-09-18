#!/usr/bin/env python3
"""Export a portable inference checkpoint with packed groupwise INT4 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llm.config import config_from_dict
from llm.model import GPT
from llm.quantization import module_storage_bytes, replace_linear_with_int4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tinystories/processed/tokenizer.json"),
    )
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
        raise SystemExit("INT4 export expects a full format-version-1 checkpoint")

    config = config_from_dict(source["config"])
    model = GPT(config.model)
    model.load_state_dict(source["model"])
    parameter_count = model.num_parameters()
    model.to(dtype=torch.bfloat16)
    quantized_modules = replace_linear_with_int4(
        model, group_size=args.group_size, exclude={"lm_head"}
    )
    storage_bytes = module_storage_bytes(model)

    bundle = {
        "format_version": 4,
        "kind": "int4-inference",
        "dtype": "bf16",
        "quantization": {
            "scheme": "symmetric-packed-int4-groupwise",
            "group_size": args.group_size,
            "scope": "transformer-linear-weights",
            "excluded": ["lm_head"],
            "modules": quantized_modules,
            "dequantize_for_linear": True,
        },
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
        "dtype": "bf16+packed-int4",
        "parameters": parameter_count,
        "group_size": args.group_size,
        "quantized_linear_modules": len(quantized_modules),
        "persistent_tensor_bytes": storage_bytes,
        "source_checkpoint": str(args.checkpoint),
        "source_step": source["step"],
        "validation_loss": source["best_val_loss"],
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported:          {args.output}")
    print(f"File size:         {args.output.stat().st_size / 2**20:.1f} MiB")
    print(f"Tensor storage:    {storage_bytes / 2**20:.1f} MiB")
    print(f"INT4 modules:      {len(quantized_modules)}")
    print(f"Group size:        {args.group_size}")
    print(f"SHA-256:           {checksum}")
    print("Excluded:          tied token embedding / LM head")
    print("Kernel:            unpack/dequantize then floating-point F.linear")


if __name__ == "__main__":
    main()
