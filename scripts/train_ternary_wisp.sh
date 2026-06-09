#!/usr/bin/env bash
# Train TinyTransformerTernary (Wisp-scale) — proper run for cuboid.
# Run from anywhere: bash scripts/train_ternary_wisp.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p ckpt logs

DATASET="${1:-data/wisp_ternary_train.txt}"
CHECKPOINT="ckpt/wisp_ternary.pt"
LOG="logs/wisp_ternary_train.log"

echo "Training Ternary Wisp"
echo "  Dataset:    $DATASET ($(wc -l < "$DATASET") lines)"
echo "  Checkpoint: $CHECKPOINT"
echo ""

nohup .venv/bin/python3 -u py/train_transformer.py \
    --file "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --arch ternary \
    --d-model 256 \
    --n-heads 4 \
    --n-layers 4 \
    --d-ff 1024 \
    --max-len 256 \
    --batch-size 64 \
    --lr 0.001 \
    --epochs 200 \
    --val-frac 0.02 \
    --patience 20 \
    --truncate \
    --status-file "logs/wisp_ternary_status.json" \
    --device cuda \
    > "$LOG" 2>&1 &

echo "Launched PID $!"
echo "  tail -f $LOG"
