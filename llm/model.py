"""A compact, reusable decoder-only GPT language model."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .attention import CausalSelfAttention
from .cache import LayerKVCache
from .config import ModelConfig
from .layers import MLP, RMSNorm, SwiGLU, matched_swiglu_hidden_size
from .sampling import sample_next_token


class TransformerBlock(nn.Module):
    """Pre-norm attention and MLP residual block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        if config.mlp_type == "swiglu":
            hidden_size = matched_swiglu_hidden_size(
                config.d_model, config.mlp_ratio, config.swiglu_multiple_of
            )
            self.mlp = SwiGLU(
                config.d_model,
                hidden_size,
                dropout=config.dropout,
                bias=config.bias,
            )
        else:
            self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cache=cache)
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

    def create_kv_caches(
        self,
        batch_size: int,
        max_length: int | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[LayerKVCache]:
        """Allocate one unexpanded K/V cache for each transformer block."""

        if max_length is None:
            max_length = self.config.context_length
        if max_length > self.config.context_length:
            raise ValueError("cache max_length cannot exceed model context_length")
        parameter = self.token_embedding.weight
        cache_device = parameter.device if device is None else device
        cache_dtype = parameter.dtype if dtype is None else dtype
        return [
            LayerKVCache(
                batch_size,
                self.config.n_kv_heads,
                max_length,
                self.config.head_dim,
                device=cache_device,
                dtype=cache_dtype,
            )
            for _ in self.blocks
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[LayerKVCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs")
        batch, sequence = input_ids.shape
        if sequence == 0:
            raise ValueError("input sequence cannot be empty")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if caches is not None and targets is not None:
            raise ValueError("cached forward is for inference and does not accept targets")

        position_offset = 0
        if caches is not None:
            if len(caches) != len(self.blocks):
                raise ValueError("one KV cache is required per transformer block")
            cache_lengths = {cache.length for cache in caches}
            if len(cache_lengths) != 1:
                raise ValueError("all layer caches must have the same active length")
            position_offset = cache_lengths.pop()
            if any(cache.batch_size != batch for cache in caches):
                raise ValueError("cache batch size must match input batch size")
        total_length = position_offset + sequence
        if total_length > self.config.context_length:
            raise ValueError(
                f"sequence length {total_length} exceeds context length "
                f"{self.config.context_length}"
            )

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(
                position_offset,
                total_length,
                device=input_ids.device,
            )
            x = x + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for index, block in enumerate(self.blocks):
            cache = caches[index] if caches is not None else None
            x = block(x, cache=cache)
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
        top_p: float | None = None,
        eos_token_id: int | None = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """Generate tokens with optional prefill and incremental K/V caching."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1] or None")

        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        if use_kv_cache and input_ids.shape[1] + max_new_tokens > self.config.context_length:
            raise ValueError(
                "cached generation prompt plus max_new_tokens cannot exceed context_length"
            )
        if max_new_tokens == 0:
            return input_ids

        generated = input_ids
        caches = None
        logits = None
        if use_kv_cache:
            device_type = input_ids.device.type
            cache_dtype = self.token_embedding.weight.dtype
            if torch.is_autocast_enabled(device_type):
                cache_dtype = torch.get_autocast_dtype(device_type)
            caches = self.create_kv_caches(
                batch_size=input_ids.shape[0],
                max_length=input_ids.shape[1] + max_new_tokens,
                device=input_ids.device,
                dtype=cache_dtype,
            )
            logits, _ = self(input_ids, caches=caches)

        for generation_step in range(max_new_tokens):
            if not use_kv_cache:
                context = generated[:, -self.config.context_length :]
                logits, _ = self(context)
            elif generation_step > 0:
                assert caches is not None
                logits, _ = self(generated[:, -1:], caches=caches)
            assert logits is not None
            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return generated

    def num_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)
