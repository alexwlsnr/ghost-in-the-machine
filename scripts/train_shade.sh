#!/usr/bin/env bash
# train_shade.sh — launch Shade (small, 10.9M params) training
#
# Shade: d=384, heads=6, layers=6, d_ff=1536, ctx=128
# Target: ~50K pairs, 25 epochs with early stopping on val loss
#
# Usage:
#   bash scripts/train_shade.sh
#   bash scripts/train_shade.sh --data data/shade_pairs.txt --output-dir checkpoints/shade
set -euo pipefail

TIER="shade"
DATA_FILE="data/training_pairs.txt"
OUTPUT_DIR="checkpoints/${TIER}"

# ── Parse optional args ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)       DATA_FILE="$2";  shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

CHECKPOINT="${OUTPUT_DIR}/best_${TIER}.pt"
LOG_DIR="logs"
STATUS_FILE="${LOG_DIR}/${TIER}_status.json"

# ── Setup directories ─────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "ERROR: training data not found at '${DATA_FILE}'" >&2
    echo "  Run: bash scripts/run_data_gen.sh" >&2
    exit 1
fi

PAIR_COUNT=$(grep -c '|' "${DATA_FILE}" 2>/dev/null || echo 0)
echo "Training Shade on ${PAIR_COUNT} pairs from ${DATA_FILE}"
echo "Checkpoint → ${CHECKPOINT}"
echo "Status     → ${STATUS_FILE}"

# ── Write initial status.json ─────────────────────────────────────────
python3 py/training_status.py --write \
    --tier "${TIER}" \
    --status-file "${STATUS_FILE}" \
    --pairs "${PAIR_COUNT}" \
    2>/dev/null || \
python3 -c "
import json, time
with open('${STATUS_FILE}', 'w') as f:
    json.dump({'tier': '${TIER}', 'status': 'starting', 'started': time.time(),
               'pairs': ${PAIR_COUNT}, 'epoch': 0, 'loss': None, 'acc': None}, f)
"

# ── Launch trainer ────────────────────────────────────────────────────
python3 py/train_transformer.py \
    --file        "${DATA_FILE}" \
    --checkpoint  "${CHECKPOINT}" \
    --d-model     384 \
    --n-heads     6 \
    --n-layers    6 \
    --d-ff        1536 \
    --max-len     128 \
    --epochs      25 \
    --lr          0.001 \
    --batch-size  128 \
    --amp \
    --qat-every   10 \
    --val-frac    0.05 \
    --patience    5 \
    --status-file "${STATUS_FILE}" \
    &

TRAIN_PID=$!
echo "Training launched (pid ${TRAIN_PID}). Watch: bash scripts/watch_training.sh ${STATUS_FILE}"
