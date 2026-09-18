#!/usr/bin/env python3
"""Overfit one fixed TinyStories batch using only LoRA adapters."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from llm.config import config_from_dict
from llm.data import TokenCorpus
from llm.lora import apply_lora, lora_state_dict, trainable_parameter_count
from llm.model import GPT
from llm.quantization import replace_linear_with_int4
from llm.training import autocast_context, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--int4-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.context_length <= 0:
        raise SystemExit("steps, batch size, and context length must be positive")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise SystemExit("LoRA overfit expects a full format-version-1 checkpoint")
    config = config_from_dict(checkpoint["config"])
    if args.context_length > config.model.context_length:
        raise SystemExit("context length exceeds model context")

    corpus = TokenCorpus(config.data.processed_dir)
    model = GPT(config.model)
    model.load_state_dict(checkpoint["model"])
    if args.qlora:
        replace_linear_with_int4(
            model, group_size=args.int4_group_size, exclude={"lm_head"}
        )
        frozen_probe = model.blocks[0].attn.q_proj.weight_packed.detach().clone()
    else:
        frozen_probe = model.blocks[0].attn.q_proj.weight.detach().clone()
    replaced = apply_lora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=0.0,
        target_modules=("q_proj", "v_proj"),
        quantized_base=args.qlora,
    )
    model.to(device).eval()  # Deterministic fixed-batch proof; gradients remain enabled.
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(args.seed)
    inputs, targets = corpus.get_batch(
        "train",
        args.batch_size,
        args.context_length,
        device,
        generator,
    )

    initial_loss = None
    final_loss = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, config.training.precision):
            _, loss = model(inputs, targets)
        assert loss is not None
        if initial_loss is None:
            initial_loss = loss.detach().float().item()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        final_loss = loss.detach().float().item()
        if step % 50 == 0 or step + 1 == args.steps:
            print(
                f"step {step:4d} | loss {final_loss:.4f} | "
                f"adapter_grad {float(grad_norm):.3f}"
            )

    assert initial_loss is not None and final_loss is not None
    first_q_proj = model.blocks[0].attn.q_proj
    frozen_after = (
        first_q_proj.base.weight_packed
        if args.qlora
        else first_q_proj.base.weight
    )
    base_unchanged = torch.equal(frozen_after.detach().cpu(), frozen_probe)
    if not base_unchanged:
        raise RuntimeError("frozen base weight changed during LoRA training")
    bundle = {
        "format_version": 1,
        "kind": "qlora-adapter" if args.qlora else "lora-adapter",
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": sha256(args.checkpoint),
        "source_step": int(checkpoint["step"]),
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "quantization": (
            {"scheme": "symmetric-packed-int4-groupwise", "group_size": args.int4_group_size}
            if args.qlora
            else None
        ),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        },
        "adapter": lora_state_dict(model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(f"Loss:                 {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"Adapted modules:      {len(replaced)}")
    print(f"Trainable parameters: {trainable_parameter_count(model):,}")
    print(f"Frozen base unchanged: {base_unchanged}")
    print(f"Adapter file:         {args.output}")
    print(f"Adapter size:         {args.output.stat().st_size / 2**20:.3f} MiB")


if __name__ == "__main__":
    main()
