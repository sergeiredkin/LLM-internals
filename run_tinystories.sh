#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/Documents/learngpt"
cd "$REPO"

if [[ -n "${LEARNGPT_ENV:-}" ]]; then
  ENV_NAME="$LEARNGPT_ENV"
elif [[ -d "$HOME/miniconda3/envs/learngpt" ]]; then
  ENV_NAME="learngpt"
else
  ENV_NAME="gpu-test"
fi

read -r -p "Story beginning [Once upon a time]: " prompt
prompt=${prompt:-Once upon a time}
read -r -p "Maximum new tokens [300]: " length
length=${length:-300}
read -r -p "Creativity 0.5-1.2 [0.8]: " temperature
temperature=${temperature:-0.8}
read -r -p "Nucleus top-p 0.5-1.0 [0.9]: " top_p
top_p=${top_p:-0.9}

if [[ -f runs/tinystories-rope-swiglu-gqa/best.pt ]]; then
  CHECKPOINT="runs/tinystories-rope-swiglu-gqa/best.pt"
else
  CHECKPOINT="runs/tinystories/best.pt"
fi

conda run -n "$ENV_NAME" python -m scripts.generate \
  --checkpoint "$CHECKPOINT" \
  --prompt "$prompt" \
  --max-new-tokens "$length" \
  --temperature "$temperature" \
  --top-k 40 \
  --top-p "$top_p" \
  --device cuda \
  --use-kv-cache
