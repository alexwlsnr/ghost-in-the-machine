#!/usr/bin/env bash
# Train Wisp+ Ternary BPE — ~20M params, d=512, 6L, ternary weights
#
# ~7MB deployed (with 8-bit embedding quantization)
# Context: 256 BPE tokens (~900 chars)
# Training data: spec512_v12_clean + all Ghost persona scenarios (184K lines)

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV=".venv/bin/python3"
DATA="data/bpe_ternary_train.txt"
TOKENIZER="data/bpe_4096.json"
CKPT="ckpt/wisp_plus_ternary_bpe.pt"
STATUS="logs/wisp_plus_ternary_bpe_status.json"
DEVICE="${1:-cuda}"

mkdir -p ckpt logs

echo "[wisp_plus] Starting Wisp+ Ternary BPE training on $DEVICE"
echo "[wisp_plus] Data: $DATA ($(wc -l < "$DATA") lines)"
echo "[wisp_plus] Checkpoint: $CKPT"

nohup $VENV -u py/train_transformer.py \
  --file "$DATA" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$CKPT" \
  --arch ternary \
  --d-model 512 \
  --n-heads 8 \
  --n-layers 6 \
  --d-ff 2048 \
  --max-len 256 \
  --batch-size 64 \
  --lr 0.001 \
  --epochs 150 \
  --val-frac 0.02 \
  --patience 20 \
  --truncate \
  --amp \
  --mask-query-loss \
  --status-file "$STATUS" \
  --device "$DEVICE" \
  > logs/wisp_plus_ternary_bpe_train.log 2>&1 &

echo "[wisp_plus] Training PID $! — tail logs/wisp_plus_ternary_bpe_train.log to follow"
