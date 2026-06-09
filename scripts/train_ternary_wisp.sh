#!/usr/bin/env bash
# Train TinyTransformerTernary (Wisp-scale) on scenarios.txt.
# Fast validation run — completes in ~30 min on RTX 3070.
# Run from repo root: bash scripts/train_ternary_wisp.sh
set -euo pipefail

DATASET="${1:-data/scenarios.txt}"
CHECKPOINT="ckpt/wisp_ternary.pt"
LOG="logs/wisp_ternary_train.log"

echo "Training Ternary Wisp"
echo "  Dataset:    $DATASET ($(wc -l < "$DATASET") lines)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Log:        $LOG"
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
    --batch-size 32 \
    --lr 0.001 \
    --epochs 30 \
    --val-frac 0.05 \
    --patience 10 \
    --truncate \
    --status-file "logs/wisp_ternary_status.json" \
    --device cuda \
    > "$LOG" 2>&1 &

echo "Launched PID $!"
echo "  tail -f $LOG"
