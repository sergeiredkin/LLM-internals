"""A compact, reusable decoder-only GPT language model."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .attention import CausalSelfAttention
from .config import ModelConfig
from .layers import MLP, RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm attention and MLP residual block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    """Decoder-only language model with learned positions or RoPE."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.vocab_size is None:
            raise ValueError("model.vocab_size must be set before constructing GPT")

        # Keep an independent immutable-style copy for checkpoints and inspection.
        self.config = replace(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (
            nn.Embedding(config.context_length, config.d_model)
            if config.position_encoding == "learned"
            else None
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._scale_residual_projections()
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs")
        batch, sequence = input_ids.shape
        if sequence == 0:
            raise ValueError("input sequence cannot be empty")
        if sequence > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence} exceeds context length "
                f"{self.config.context_length}"
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(sequence, device=input_ids.device)
            x = x + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * sequence, self.config.vocab_size),
                targets.reshape(batch * sequence),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens without a KV cache; caching is a later milestone."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")

        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.context_length :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, next_logits.shape[-1])
                threshold = torch.topk(next_logits, k).values[:, -1, None]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return generated

    def num_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)
