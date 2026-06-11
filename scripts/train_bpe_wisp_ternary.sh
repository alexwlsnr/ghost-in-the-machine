#!/usr/bin/env bash
# Train BPE Ternary Wisp — same arch as BPE fp32 but {-1,0,+1} weights
#
# Training data: spec512_v12_clean + all Ghost persona scenarios (184K lines)
# Architecture: d=320, 4L, 4H, ff=1280, ctx=256, vocab=4099, ternary weights
# ~7.6M params → ~1.9MB deployed (ternary ~2 bits/weight)

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV=".venv/bin/python3"
DATA="data/bpe_ternary_train.txt"
TOKENIZER="data/bpe_4096.json"
CKPT="ckpt/wisp_bpe_ternary.pt"
STATUS="logs/wisp_bpe_ternary_status.json"
DEVICE="${1:-cuda}"

mkdir -p ckpt logs

if [ ! -f "$TOKENIZER" ]; then
  echo "[bpe_ternary] Tokenizer not found at $TOKENIZER — run train_bpe.py first"
  exit 1
fi

echo "[bpe_ternary] Starting BPE Ternary Wisp training on $DEVICE"
echo "[bpe_ternary] Data: $DATA ($(wc -l < "$DATA") lines)"
echo "[bpe_ternary] Checkpoint: $CKPT"

nohup $VENV -u py/train_transformer.py \
  --file "$DATA" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$CKPT" \
  --arch ternary \
  --d-model 320 \
  --n-heads 4 \
  --n-layers 4 \
  --d-ff 1280 \
  --max-len 256 \
  --batch-size 64 \
  --lr 0.001 \
  --epochs 150 \
  --val-frac 0.02 \
  --patience 15 \
  --truncate \
  --amp \
  --mask-query-loss \
  --status-file "$STATUS" \
  --device "$DEVICE" \
  > logs/wisp_bpe_ternary_train.log 2>&1 &

echo "[bpe_ternary] Training PID $! — tail logs/wisp_bpe_ternary_train.log to follow"
