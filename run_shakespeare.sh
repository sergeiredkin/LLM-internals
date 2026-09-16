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

read -r -p "Character name [ROMEO]: " character
character=${character:-ROMEO}
read -r -p "Number of new characters [600]: " length
length=${length:-600}

prompt=$(printf '%s:\n' "$character")
conda run -n "$ENV_NAME" python -m scripts.generate \
  --checkpoint runs/shakespeare/best.pt \
  --prompt "$prompt" \
  --max-new-tokens "$length" \
  --temperature 0.8 \
  --top-k 30 \
  --device cuda
