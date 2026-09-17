# learnGPT — LLM Internals From First Principles

An educational project that starts with isolated PyTorch demonstrations of core LLM
mechanisms and assembles them into a working decoder-only language model.

The first milestone is complete: a 10.7M-parameter character GPT trained from scratch on
Tiny Shakespeare using a local NVIDIA RTX 3060 12 GB.

## Stage 1 result

| Item | Result |
|---|---:|
| Architecture | 6 layers, width 384, 6 heads, context 256 |
| Parameters | 10,745,088 |
| Vocabulary | 65 characters |
| Precision | BF16 |
| Best checkpoint | Step 2,250 |
| Best validation loss | **1.4910** |
| Peak PyTorch VRAM | **0.47 GiB reserved** |
| Typical throughput | 75k–83k tokens/second |

The model learned speaker structure, punctuation, line breaks, and Shakespeare-like prose:

```text
ROMEO:
What is the world, when the sea miserable,
Or that rude the gates such at him with his bard,
As was, away, when he dares him hence,
Let forth him that now incannot weep.
```

The text is generated rather than copied, so it resembles Shakespeare without being
factually or grammatically reliable.

## Components

- Causal self-attention through PyTorch SDPA
- RMSNorm and pre-norm residual blocks
- GELU MLP
- Learned positional embeddings
- Tied token/output embeddings
- Character tokenizer and memory-mapped corpus
- AdamW with warmup and cosine decay
- BF16/FP16 mixed precision
- Gradient accumulation and clipping
- Atomic checkpoints and resume support
- Validation, generation, and internal model inspection

The numbered root scripts remain small experiments explaining attention, AdamW, KV
caching, LoRA, Flash Attention, quantization, sampling, cross-entropy, backpropagation,
and RMSNorm. Reusable model code lives under `llm/`.

## Environment

```bash
conda env create -f environment.yml
conda activate learngpt
python check_environment.py
```

An existing local environment can also be used if it has a CUDA-enabled PyTorch build.

## Prepare the corpus

Tiny Shakespeare is included with source and checksum information in
`data/shakespeare/README.md`.

```bash
python -m scripts.prepare_data --config configs/shakespeare.yaml
```

## Test

```bash
python -m unittest discover -s tests -v
python -m scripts.overfit_batch
```

## Train

```bash
python -m scripts.train --config configs/shakespeare.yaml
```

Training outputs are written to `runs/` and intentionally excluded from Git. The trainer
refuses to start with less than 8 GiB free unless `--allow-shared-gpu` is explicitly used.

## Generate

```bash
python -m scripts.generate \
  --checkpoint runs/shakespeare/best.pt \
  --prompt $'ROMEO:\n' \
  --max-new-tokens 600 \
  --temperature 0.8 \
  --top-k 30
```

For an interactive shortcut:

```bash
./run_shakespeare.sh
```

## Inspect model internals

```bash
python -m scripts.inspect_model \
  --checkpoint runs/shakespeare/best.pt \
  --text $'ROMEO:\nWhat is the world?'
```

The inspector displays token IDs, tensor shapes, per-block activation statistics,
next-token probabilities, parameter allocation, teacher-forced loss, and gradient norms.

## Reproducible evaluation and export

Architecture experiments use fixed validation windows and a fixed generation prompt/seed:

```bash
python -m scripts.evaluate \
  --checkpoint runs/tinystories/best.pt \
  --output reports/results/tinystories-learned-baseline.json

python -m scripts.export_inference \
  --checkpoint runs/tinystories/best.pt \
  --output exports/tinystories-29m-bf16.pt
```

The publishable baseline is in `reports/tinystories-baseline.md`; structured comparison rows
are stored in `reports/experiments.csv`. Inference exports are excluded from Git and can be
attached to GitHub Releases.

## First findings

1. A one-batch overfit test is the fastest end-to-end correctness check. Loss fell from
   4.23 to effectively zero, proving that data shifting, masking, loss, gradients, and
   optimizer updates agree.
2. Initial loss was close to `ln(65)`, as expected for a random 65-character model.
3. Validation loss improved from 4.26 to 1.49, while later training reduced training loss
   but worsened validation loss. Keeping a best-validation checkpoint mattered.
4. Gradient clipping handled early spikes without destabilizing training.
5. The model consumed much less memory than the 12 GB budget: only 0.47 GiB reserved at
   the selected micro-batch size.
6. Character models can learn convincing local form with little data, but subword tokens
   are the logical next step for broader language and domain learning.

Detailed results are recorded in `Project/stage-1-shakespeare.md`.

## Stage 2 data preparation

The TinyStories pipeline downloads checksum-verified official Parquet shards, preserves
documents as JSONL, trains an 8K byte-level BPE tokenizer, and writes compact token
streams:

```bash
python -m scripts.prepare_tinystories
python -m scripts.inspect_tokenizer
python -m scripts.overfit_batch --config configs/tinystories.yaml
```

The current subset contains 50,000 training documents (10.97M BPE tokens) and 5,000
official validation documents (0.99M tokens). Raw and generated data are excluded from Git.

The 29.5M-parameter token model trained for 5,000 steps in 33 minutes while sharing the GPU
with Ollama. It reached **1.8759 validation loss** and generates coherent multi-paragraph
children's stories. Detailed metrics are in `Project/stage-2-tinystories.md`.

## Stage 3 controlled architecture experiments

Replacing learned positions with RoPE reduced the model by 262,144 parameters and improved the
fixed validation loss from **1.9026 to 1.8401** (perplexity 6.70 to 6.30). Dataset, seed,
architecture dimensions, optimizer, and 5,000-step budget were held constant. See
`reports/tinystories-rope-ablation.md` for the protocol, limitations, and generated sample.

A parameter-matched SwiGLU experiment then reached **1.8316 loss** and **6.24 perplexity** with
65,536 fewer parameters than RoPE + GELU. The gain was small, inference was about 11% slower, and
the fixed generated sample was less coherent. See `reports/tinystories-swiglu-ablation.md`.

## Roadmap

1. **Done:** character-level Tiny Shakespeare GPT
2. **Done:** token-level TinyStories model with an 8K BPE vocabulary
3. **In progress:** RoPE and SwiGLU done; next are GQA, KV cache, and top-p sampling
4. Licensed petroleum corpus, domain evaluation, RAG, and optional QLoRA adaptation
