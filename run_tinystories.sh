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

conda run -n "$ENV_NAME" python -m scripts.generate \
  --checkpoint runs/tinystories/best.pt \
  --prompt "$prompt" \
  --max-new-tokens "$length" \
  --temperature "$temperature" \
  --top-k 40 \
  --device cuda
