#!/usr/bin/env bash
# train_specter.sh — launch Specter (large, 57.3M params) training
#
# Specter: d=768, heads=12, layers=8, d_ff=3072, ctx=256
# Target: ~200K pairs, 12 epochs with early stopping on val loss
#
# NOTE: Specter at float32 (219 MB) cannot be committed to git (>100 MB limit).
#       Quantize to 8-bit (~57 MB) or 4-bit (~28 MB) before shipping.
#       See docs/multi-model-plan.md — Phase 5 (quantization path).
#
# NOTE: Consider respeccing to d=512/L=8 (~26M params, ~13 MB at 4-bit)
#       before committing to d=768 — see the open decision in the plan.
#
# Usage:
#   bash scripts/train_specter.sh
#   bash scripts/train_specter.sh --data data/specter_pairs.txt --output-dir checkpoints/specter
set -euo pipefail

TIER="specter"
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
echo "Training Specter on ${PAIR_COUNT} pairs from ${DATA_FILE}"
echo "Checkpoint → ${CHECKPOINT}"
echo "Status     → ${STATUS_FILE}"
echo ""
echo "REMINDER: Specter float32 (219 MB) cannot be git-committed."
echo "          Quantize before shipping (see Phase 5 in plan)."
echo ""

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
    --d-model     768 \
    --n-heads     12 \
    --n-layers    8 \
    --d-ff        3072 \
    --max-len     256 \
    --epochs      12 \
    --lr          0.0005 \
    --batch-size  64 \
    --amp \
    --qat-every   8 \
    --val-frac    0.05 \
    --patience    3 \
    --status-file "${STATUS_FILE}" \
    &

TRAIN_PID=$!
echo "Training launched (pid ${TRAIN_PID}). Watch: bash scripts/watch_training.sh ${STATUS_FILE}"
