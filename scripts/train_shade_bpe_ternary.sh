#!/usr/bin/env bash
# Train Shade BPE Ternary — ~11M params, d=512, 4L, ternary weights, mixed case
#
# Deployed size: ~22MB BF16 equivalent (4099×512 BPE embed + ternary weights)
# Context: 256 BPE tokens (~900 chars)
# Training data: shade_bpe_train.txt (302K lines — UltraChat, SmolTalk, OASST2,
#                DailyDialog, ProSocial, Persona-Chat + Ghost scenarios)
# Note: --preserve-case — first model in stack trained on natural mixed-case text.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV=".venv/bin/python3"
DATA="data/shade_bpe_train.txt"
TOKENIZER="data/bpe_4096.json"
CKPT="ckpt/shade_bpe_ternary.pt"
STATUS="logs/shade_bpe_ternary_status.json"
DEVICE="${1:-cuda}"

mkdir -p ckpt logs

echo "[shade_bpe_ternary] Starting Shade BPE Ternary training on $DEVICE"
echo "[shade_bpe_ternary] Data: $DATA ($(wc -l < "$DATA") lines)"
echo "[shade_bpe_ternary] Checkpoint: $CKPT"

nohup $VENV -u py/train_transformer.py \
  --file "$DATA" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$CKPT" \
  --arch ternary \
  --d-model 512 \
  --n-heads 8 \
  --n-layers 4 \
  --d-ff 2048 \
  --max-len 256 \
  --batch-size 64 \
  --lr 0.0008 \
  --epochs 100 \
  --val-frac 0.02 \
  --patience 20 \
  --truncate \
  --amp \
  --preserve-case \
  --mask-query-loss \
  --status-file "$STATUS" \
  --device "$DEVICE" \
  > logs/shade_bpe_ternary_train.log 2>&1 &

echo "[shade_bpe_ternary] Training PID $! — tail logs/shade_bpe_ternary_train.log to follow"
