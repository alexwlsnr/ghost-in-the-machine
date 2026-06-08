#!/usr/bin/env bash
# run_data_gen.sh — generate training pairs via expand_data.py
#
# Calls the teacher model (gemma4-e4b-distill via llama-swap on :8080)
# to expand 200 seed templates into ~50K prompt/response pairs.
#
# Output: data/training_pairs.txt (pipe-separated Q|R lines)
#
# Prerequisites:
#   - llama-swap running on http://localhost:8080/v1
#   - gemma4-e4b-distill loaded (or set EXPAND_MODEL / EXPAND_ENDPOINT)
#   - data/templates.txt exists (already committed)
#
# Usage:
#   bash scripts/run_data_gen.sh
#   EXPAND_MODEL=llama3.2-3b bash scripts/run_data_gen.sh   # override model
set -euo pipefail

MODEL="${EXPAND_MODEL:-gemma4-e4b-distill}"
ENDPOINT="${EXPAND_ENDPOINT:-http://localhost:8080/v1}"
OUTPUT="data/training_pairs.txt"
TEMPLATES="data/templates.txt"
CHECKPOINT="data/expand_checkpoint.json"

# ── Sanity checks ─────────────────────────────────────────────────────
if [[ ! -f "${TEMPLATES}" ]]; then
    echo "ERROR: templates file not found at '${TEMPLATES}'" >&2
    exit 1
fi

if [[ ! -f "py/expand_data.py" ]]; then
    echo "ERROR: py/expand_data.py not found" >&2
    echo "  This script is part of Phase 2 (see docs/multi-model-plan.md)." >&2
    exit 1
fi

TEMPLATE_COUNT=$(grep -v '^#' "${TEMPLATES}" | grep -v '^$' | wc -l)
echo "Data generation starting"
echo "  Templates : ${TEMPLATE_COUNT} (from ${TEMPLATES})"
echo "  Model     : ${MODEL}"
echo "  Endpoint  : ${ENDPOINT}"
echo "  Output    : ${OUTPUT}"
echo "  Checkpoint: ${CHECKPOINT}"
echo ""

mkdir -p data

# ── Launch expand_data.py ─────────────────────────────────────────────
python3 py/expand_data.py \
    --output                "${OUTPUT}" \
    --model                 "${MODEL}" \
    --endpoint              "${ENDPOINT}" \
    --workers               32 \
    --max-prompts           50000 \
    --max-ctx               256 \
    --samples-per-template  5 \
    --templates             "${TEMPLATES}" \
    --checkpoint            "${CHECKPOINT}"
