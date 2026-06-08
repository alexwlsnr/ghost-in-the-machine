#!/usr/bin/env bash
# train_wraith.sh — Wraith Linux-guru model MVP training script
#
# Uses Wisp-scale architecture for rapid iteration (3-4K pairs, ~60 epochs).
# ctx=128 (larger than Wisp's 64) to handle technical content like paths and flags.
# --preserve-case is critical: commands and flags must NOT be uppercased.
#
# Usage:
#   bash scripts/train_wraith.sh                     # train with defaults
#   bash scripts/train_wraith.sh --epochs 30         # quick smoke-test
#
# Prerequisites:
#   1. Generate training data first:
#      python3 py/wraith_data.py --output data/wraith_training_pairs.txt
#   2. Activate venv if needed:
#      source .venv/bin/activate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA="data/wraith_training_pairs.txt"
CHECKPOINT="ckpt/wraith_mvp.pt"
STATUS="logs/wraith_status.json"

# ─── Guards ─────────────────────────────────────────────────────────────────
if [[ ! -f "$DATA" ]]; then
    echo "ERROR: Training data not found at $DATA"
    echo "Generate it first:"
    echo "  python3 py/wraith_data.py --output $DATA"
    exit 1
fi

PAIR_COUNT=$(grep -c '|' "$DATA" 2>/dev/null || echo 0)
echo "Training data: $DATA ($PAIR_COUNT pairs)"

# ─── Directories ────────────────────────────────────────────────────────────
mkdir -p ckpt logs

# ─── Hyperparameters (Wisp-scale, ctx=128, preserve-case) ───────────────────
# Architecture: d=256, 4 heads, 4 layers, d_ff=1024 (~3.3M params)
# ctx=128: double Wisp's 64 — paths and flags need the extra room
# epochs=60: enough for 3-4K pairs; adjust down if overfitting kicks in early
# patience=8: early-stop if val loss stagnates
# val-frac=0.10: hold out 10% for honest early-stopping

python3 py/train_transformer.py \
    --file "$DATA" \
    --d-model 256 \
    --n-heads 4 \
    --n-layers 4 \
    --d-ff 1024 \
    --max-len 128 \
    --epochs 60 \
    --lr 0.003 \
    --batch-size 64 \
    --amp \
    --qat-every 5 \
    --val-frac 0.10 \
    --patience 8 \
    --checkpoint "$CHECKPOINT" \
    --status-file "$STATUS" \
    --preserve-case \
    "$@"

echo ""
echo "Checkpoint saved: $CHECKPOINT"
echo "Status:           $STATUS"
echo ""
echo "Next steps:"
echo "  1. Run eval: python3 test/eval_wraith.py --checkpoint $CHECKPOINT"
echo "  2. If >=15/20 on eval set, proceed to Wraith-C (d=384, 8 layers)"
