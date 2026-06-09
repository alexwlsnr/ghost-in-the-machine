#!/usr/bin/env bash
# Train Spec512 v1.2 on the clean v1.2 dataset.
# Run from repo root: bash scripts/train_spec512_v12.sh
set -euo pipefail

DATASET="data/spec512_v12_clean.txt"
CHECKPOINT="ckpt/spec512_v1.2.pt"
LOG="logs/spec512_v12_train.log"

if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET"
    echo "Run: python3 py/build_v12_dataset.py"
    exit 1
fi

echo "Training Spec512 v1.2"
echo "  Dataset:    $DATASET ($(wc -l < "$DATASET") lines)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Log:        $LOG"
echo ""

nohup .venv/bin/python3 py/train_transformer.py \
    --file "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --arch classic \
    --d-model 512 \
    --n-heads 8 \
    --n-layers 8 \
    --d-ff 2048 \
    --max-len 1024 \
    --batch-size 64 \
    --lr 0.0005 \
    --epochs 100 \
    --val-frac 0.02 \
    --patience 15 \
    --device cuda \
    > "$LOG" 2>&1 &

echo "Launched PID $!"
echo "  tail -f $LOG"
