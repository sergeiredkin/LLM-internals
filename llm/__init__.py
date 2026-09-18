"""Reusable components for the learnGPT training project."""

from .attention import repeat_kv
from .cache import LayerKVCache
from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    config_from_dict,
    load_config,
)
from .bsee import BSEEProduction
from .data import TokenCorpus, prepare_char_corpus
from .eia import (
    EIAObservation,
    change,
    connect,
    insert_observations,
    latest_value,
    observation_from_record,
    percent_change,
    series_values,
    year_over_year,
)
from .layers import SwiGLU, matched_swiglu_hidden_size
from .lora import (
    LoRALinear,
    apply_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    trainable_parameter_count,
)
from .model import GPT
from .petroleum_quality import quality_flags
from .rag import (
    BM25Retriever,
    Chunk,
    ContextResult,
    Document,
    RetrievalResult,
    chunk_document,
    load_jsonl_documents,
    retrieval_metrics,
    tokenize,
    write_jsonl_documents,
)
from .quantization import (
    Int4Linear,
    Int8Linear,
    dequantize_int4_groupwise,
    dequantize_int8_per_channel,
    pack_signed_int4,
    quantize_int4_groupwise,
    quantize_int8_per_channel,
    replace_linear_with_int4,
    replace_linear_with_int8,
    unpack_signed_int4,
)
from .rope import RotaryEmbedding, rotate_half
from .sampling import sample_next_token, top_k_filter, top_p_filter
from .tokenizer import BPETokenizer, CharTokenizer, load_tokenizer, tokenizer_from_json

__all__ = [
    "GPT",
    "BSEEProduction",
    "EIAObservation",
    "connect",
    "observation_from_record",
    "insert_observations",
    "latest_value",
    "series_values",
    "year_over_year",
    "change",
    "percent_change",
    "quality_flags",
    "Document",
    "Chunk",
    "ContextResult",
    "RetrievalResult",
    "BM25Retriever",
    "chunk_document",
    "load_jsonl_documents",
    "write_jsonl_documents",
    "retrieval_metrics",
    "tokenize",
    "SwiGLU",
    "matched_swiglu_hidden_size",
    "LoRALinear",
    "apply_lora",
    "merge_lora",
    "lora_state_dict",
    "load_lora_state_dict",
    "trainable_parameter_count",
    "RotaryEmbedding",
    "rotate_half",
    "repeat_kv",
    "LayerKVCache",
    "Int8Linear",
    "Int4Linear",
    "quantize_int8_per_channel",
    "quantize_int4_groupwise",
    "dequantize_int4_groupwise",
    "pack_signed_int4",
    "unpack_signed_int4",
    "replace_linear_with_int4",
    "dequantize_int8_per_channel",
    "replace_linear_with_int8",
    "sample_next_token",
    "top_k_filter",
    "top_p_filter",
    "CharTokenizer",
    "BPETokenizer",
    "load_tokenizer",
    "tokenizer_from_json",
    "TokenCorpus",
    "prepare_char_corpus",
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "config_from_dict",
    "load_config",
]
