#!/usr/bin/env bash
# train_shade.sh — Launch Shade training as a detached background job.
#
# Usage:
#   bash scripts/train_shade.sh [--data FILE] [--workers N]
#
# The job runs under nohup in the background. Logs go to logs/shade_train.log.
# Monitor progress with:
#   bash scripts/watch_training.sh logs/shade_status.json
# Or tail the log:
#   tail -f logs/shade_train.log

set -euo pipefail

# ─── Defaults (override with env vars or flags) ───────────────────────────────
DATA="data/training_pairs.txt"
CKPT="ckpt/shade_fp32.pt"
STATUS="logs/shade_status.json"
LOG="logs/shade_train.log"
EPOCHS=30

# Parse simple --data / --workers flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)    DATA="$2";    shift 2 ;;
        --workers) : ;           shift 2 ;;   # reserved for future use
        --epochs)  EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

mkdir -p ckpt logs

echo "Shade training launcher"
echo "  data:       $DATA"
echo "  checkpoint: $CKPT"
echo "  status:     $STATUS"
echo "  log:        $LOG"
echo "  epochs:     $EPOCHS"

# ─── Validate data file ───────────────────────────────────────────────────────
if [[ ! -f "$DATA" ]]; then
    echo "Error: training data file not found: $DATA" >&2
    echo "Generate it first (see docs/multi-model-plan.md, Phase 2)." >&2
    exit 1
fi

# ─── Write initial status ─────────────────────────────────────────────────────
python3 py/training_status.py --write "$STATUS" \
    --tier shade \
    --phase training \
    --state running \
    --epochs-total "$EPOCHS" \
    --checkpoint "$CKPT"

echo "Initial status written to $STATUS"

# ─── Launch detached training job ────────────────────────────────────────────
# Shade architecture: d=384, heads=6, layers=6, ff=1536, ctx=128
# lr=0.001, batch=128, amp, qat-every=10, patience=5 (early stopping)
nohup .venv/bin/python3 py/train_transformer.py \
    --file "$DATA" \
    --d-model 384 \
    --n-heads 6 \
    --n-layers 6 \
    --d-ff 1536 \
    --max-len 128 \
    --epochs "$EPOCHS" \
    --lr 0.001 \
    --batch-size 128 \
    --amp \
    --qat-every 10 \
    --val-frac 0.05 \
    --patience 5 \
    --checkpoint "$CKPT" \
    --status-file "$STATUS" \
    > "$LOG" 2>&1 &

TRAIN_PID=$!
echo "Shade training launched (pid ${TRAIN_PID})."
echo "Watch: bash scripts/watch_training.sh ${STATUS}"
echo "Log:   tail -f ${LOG}"

# Update status with the actual PID
python3 py/training_status.py --write "$STATUS" --pid "$TRAIN_PID"
