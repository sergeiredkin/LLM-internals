#!/usr/bin/env python3
"""Inspect tokens, activations, predictions, loss, and gradients in a checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from llm.config import config_from_dict
from llm.data import TokenCorpus
from llm.model import GPT
from llm.training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/shakespeare/best.pt"))
    parser.add_argument("--text", help="text to inspect; prompts interactively when omitted")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    return {
        "mean": values.mean().item(),
        "std": values.std().item(),
        "rms": values.square().mean().sqrt().item(),
        "max": values.abs().max().item(),
    }


def gradient_norm(module: torch.nn.Module) -> float:
    squared = sum(
        parameter.grad.detach().float().square().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    text = args.text if args.text is not None else input("Text [ROMEO:\\n]: ")
    if not text:
        text = "ROMEO:\n"

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_dict(checkpoint["config"])
    corpus = TokenCorpus(config.data.processed_dir)
    tokenizer = corpus.tokenizer
    try:
        encoded = tokenizer.encode(text)
    except ValueError as error:
        raise SystemExit(f"Text cannot be encoded: {error}") from error

    if len(encoded) > config.model.context_length:
        print(
            f"Input has {len(encoded)} tokens; keeping the last "
            f"{config.model.context_length}."
        )
        encoded = encoded[-config.model.context_length :]
        text = tokenizer.decode(encoded)

    model = GPT(config.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

    print(f"Checkpoint: {args.checkpoint} (step {checkpoint['step']})")
    print(f"Device: {device} | Parameters: {model.num_parameters():,}")
    print(f"Text: {text!r}")
    print(f"Token IDs: {encoded}")
    print("Token mapping:")
    print("  " + " ".join(f"{token_id}:{tokenizer.token_for_id(token_id)!r}" for token_id in encoded))
    print("\nShapes:")
    print(f"  input                 {tuple(input_ids.shape)}")
    print(f"  token embeddings      {(1, len(encoded), config.model.d_model)}")
    print(
        f"  Q per block           {(1, config.model.n_heads, len(encoded), config.model.head_dim)}"
    )
    print(
        f"  K/V per block         {(1, config.model.n_kv_heads, len(encoded), config.model.head_dim)}"
    )
    print(f"  attention scores      {(1, config.model.n_heads, len(encoded), len(encoded))}")
    print(f"  logits                {(1, len(encoded), config.model.vocab_size)}")

    activations: dict[int, dict[str, float]] = {}
    hooks = []
    for index, block in enumerate(model.blocks):
        def capture(_module, _inputs, output, layer=index):
            activations[layer] = tensor_stats(output)
        hooks.append(block.register_forward_hook(capture))

    with torch.no_grad():
        logits, _ = model(input_ids)
    for hook in hooks:
        hook.remove()

    print("\nBlock output statistics:")
    for index in range(len(model.blocks)):
        stats = activations[index]
        print(
            f"  block {index:2d}: mean={stats['mean']:+.4f} std={stats['std']:.4f} "
            f"rms={stats['rms']:.4f} max|x|={stats['max']:.4f}"
        )

    probabilities = logits[0, -1].float().softmax(dim=-1)
    k = min(args.top_k, tokenizer.vocab_size)
    values, indices = probabilities.topk(k)
    print(f"\nTop {k} predictions after {text!r}:")
    for probability, token_id in zip(values.tolist(), indices.tolist()):
        print(f"  {tokenizer.token_for_id(token_id)!r:18s} {probability * 100:6.2f}%")

    print("\nParameter allocation:")
    print(f"  token embedding/head  {parameter_count(model.token_embedding):>10,} (tied)")
    print(f"  position embedding    {parameter_count(model.position_embedding):>10,}")
    print(f"  transformer blocks    {parameter_count(model.blocks):>10,}")
    print(f"  final norm            {parameter_count(model.final_norm):>10,}")

    if len(encoded) < 2:
        print("\nEnter at least two characters to inspect loss and gradients.")
        return

    model.zero_grad(set_to_none=True)
    shifted_inputs = input_ids[:, :-1]
    shifted_targets = input_ids[:, 1:]
    _, loss = model(shifted_inputs, shifted_targets)
    assert loss is not None
    loss.backward()
    print(f"\nTeacher-forced loss on this text: {loss.item():.4f}")
    print("Gradient norms:")
    print(f"  token embedding       {gradient_norm(model.token_embedding):.6f}")
    for index, block in enumerate(model.blocks):
        print(f"  block {index:2d}              {gradient_norm(block):.6f}")
    print(f"  final norm            {gradient_norm(model.final_norm):.6f}")
    print(f"  whole model           {gradient_norm(model):.6f}")


if __name__ == "__main__":
    main()
