"""Reusable logit filtering and token sampling primitives."""

from __future__ import annotations

import torch


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep exactly the highest ``top_k`` logits in each row."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocabulary)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    k = min(top_k, logits.shape[-1])
    values, indices = torch.topk(logits, k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, indices, values)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest high-probability set whose mass reaches ``top_p``.

    Probabilities are computed after sorting logits from most to least likely. The token that
    crosses the cumulative-probability threshold is retained, so every row keeps at least one
    finite logit.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocabulary)")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_p == 1.0:
        return logits.clone()

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    # Keep a token while the mass before it is below p. This includes the token that
    # reaches/crosses p, but excludes another token when the preceding mass is exactly p.
    keep = cumulative - sorted_probabilities < top_p
    sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, sorted_indices, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row after temperature, top-k, and top-p filtering."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocabulary)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1] or None")

    scaled = logits / temperature
    if top_k == 1:
        return scaled.argmax(dim=-1, keepdim=True)
    if top_k is not None:
        scaled = top_k_filter(scaled, top_k)
    if top_p is not None:
        scaled = top_p_filter(scaled, top_p)
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)
