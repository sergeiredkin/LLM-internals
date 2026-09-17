#!/usr/bin/env python3
"""Overfit one fixed batch to verify the complete learning path."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from llm.config import load_config
from llm.data import TokenCorpus
from llm.model import GPT
from llm.training import create_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/shakespeare.yaml"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--target-loss", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.target_loss <= 0:
        raise SystemExit("--steps and --target-loss must be positive")

    experiment = load_config(args.config)
    corpus = TokenCorpus(experiment.data.processed_dir)
    # Keep the architecture identical in kind but small enough for a quick CPU proof.
    model_config = replace(
        experiment.model,
        vocab_size=corpus.tokenizer.vocab_size,
        context_length=64,
        n_layers=2,
        d_model=128,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    training_config = replace(
        experiment.training,
        device="cpu",
        precision="fp32",
        learning_rate=3e-3,
        min_learning_rate=3e-3,
        weight_decay=0.0,
        grad_clip=1.0,
    )

    torch.manual_seed(training_config.seed)
    model = GPT(model_config)
    optimizer = create_optimizer(model, training_config)
    generator = torch.Generator().manual_seed(training_config.seed)
    inputs, targets = corpus.get_batch(
        "train", batch_size=4, context_length=model_config.context_length, generator=generator
    )

    initial_loss = None
    final_loss = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = loss.item()
        if step % 50 == 0 or step == args.steps - 1:
            print(
                f"step {step:3d} | loss {final_loss:.4f} | "
                f"grad {float(gradient_norm):.3f}"
            )

    assert initial_loss is not None and final_loss is not None
    print(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")
    if final_loss >= args.target_loss:
        raise SystemExit(
            f"FAILED: final loss {final_loss:.4f} did not reach {args.target_loss:.4f}"
        )
    print("PASS: one fixed batch was memorized; forward, loss, backward, and update agree.")


if __name__ == "__main__":
    main()
