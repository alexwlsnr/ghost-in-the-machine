#!/usr/bin/env bash
# GQA A/B spike: Wisp byte-level ternary_modern with n_kv_heads=2 (vs the MHA
# baseline at n_kv_heads=4, val 1.3266). IDENTICAL data/tokenizer/config — only
# the KV-head count differs — so the val-loss delta isolates GQA's cost.
# n_kv_heads=2 on 4 heads = 2× KV-cache reduction (for Spectre's 8 heads it'd be 4×).

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV=".venv/bin/python3"
DATA="data/bpe_ternary_train.txt"
TOKENIZER="data/bpe_bytelevel_4099.json"
CKPT="ckpt/wisp_gqa_ternary_modern.pt"
STATUS="logs/wisp_gqa_status.json"
DEVICE="${1:-cuda}"

mkdir -p ckpt logs
echo "[wisp_gqa] Starting GQA (n_kv_heads=2) A/B on $DEVICE"
echo "[wisp_gqa] Data: $DATA ($(wc -l < "$DATA") lines)  Tokenizer: $TOKENIZER"

nohup $VENV -u py/train_transformer.py \
  --file "$DATA" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$CKPT" \
  --arch ternary_modern \
  --d-model 320 --n-heads 4 --n-kv-heads 2 --n-layers 4 --d-ff 1280 \
  --max-len 256 --batch-size 64 --lr 0.001 \
  --epochs 150 --val-frac 0.02 --patience 15 \
  --truncate --amp --mask-query-loss \
  --status-file "$STATUS" \
  --device "$DEVICE" \
  > logs/wisp_gqa_train.log 2>&1 &

echo "[wisp_gqa] Training PID $! — tail logs/wisp_gqa_train.log"
