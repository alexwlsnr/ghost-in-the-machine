#!/usr/bin/env bash
# Train BPE Wisp — 6.3M param byte-pair-encoding transformer
#
# Architecture: d=320, 4L, 4H, ff=1280, ctx=256 BPE tokens (~768-1024 chars)
# ~6.3MB at 8-bit — fits in the 7MB "new Wisp" target
#
# Step 1: train BPE tokenizer (one-time, ~5-10 min on 10MB of text)
# Step 2: train model with BPE tokenizer (fast — ~8 min on 5080 for 150 epochs)

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV=".venv/bin/python3"
DATA="data/spec512_v12_clean.txt"
TOKENIZER="data/bpe_4096.json"
CKPT="ckpt/wisp_bpe.pt"
STATUS="logs/wisp_bpe_status.json"
DEVICE="${1:-cuda}"

mkdir -p ckpt logs

# ── Step 1: tokenizer ───────────────────────────────────────────────────────
if [ -f "$TOKENIZER" ]; then
  echo "[bpe_wisp] Tokenizer already exists at $TOKENIZER — skipping training"
  echo "[bpe_wisp] Delete it to retrain: rm $TOKENIZER"
else
  echo "[bpe_wisp] Training BPE tokenizer (vocab=4096, 10M chars from $DATA)..."
  $VENV py/train_bpe.py \
    --data "$DATA" \
    --vocab-size 4096 \
    --out "$TOKENIZER" \
    --max-chars 10000000
fi

# ── Step 2: model ────────────────────────────────────────────────────────────
echo "[bpe_wisp] Starting model training on $DEVICE"
echo "[bpe_wisp] Checkpoint: $CKPT"

nohup $VENV -u py/train_transformer.py \
  --file "$DATA" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$CKPT" \
  --arch classic \
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
  > logs/wisp_bpe_train.log 2>&1 &

echo "[bpe_wisp] Training PID $! — tail logs/wisp_bpe_train.log to follow"
