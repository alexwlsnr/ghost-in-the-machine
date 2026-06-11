#!/usr/bin/env bash
# Self-restarting training manager for BPE Ternary Wisp.
# Handles repeated CUDA timeout crashes by resuming from checkpoint.
# Runs until training reports 'done' or 'early_stopped'.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV=".venv/bin/python3"
DATA="data/bpe_ternary_train.txt"
TOKENIZER="data/bpe_4096.json"
CKPT="ckpt/wisp_bpe_ternary.pt"
STATUS="logs/wisp_bpe_ternary_status.json"

mkdir -p ckpt logs

is_done() {
    python3 -c "
import json, sys
try:
    d = json.load(open('$STATUS'))
    sys.exit(0 if d.get('state') in ('done','early_stopped') else 1)
except: sys.exit(1)
" 2>/dev/null
}

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[managed] Attempt $attempt — $(date -u '+%H:%M:%S UTC')"

    if is_done; then
        echo "[managed] Training complete. Exiting."
        break
    fi

    # Use --resume if checkpoint exists
    RESUME_FLAG=""
    if [ -f "$CKPT" ]; then
        RESUME_FLAG="--resume $CKPT"
    fi

    $VENV -u py/train_transformer.py \
        --file "$DATA" \
        --tokenizer "$TOKENIZER" \
        $RESUME_FLAG \
        --checkpoint "$CKPT" \
        --arch ternary \
        --d-model 320 --n-heads 4 --n-layers 4 --d-ff 1280 \
        --max-len 256 --batch-size 16 --lr 0.001 \
        --epochs 150 --val-frac 0.02 --patience 15 \
        --truncate --amp --mask-query-loss \
        --status-file "$STATUS" \
        --device cuda 2>&1 || true

    if is_done; then
        echo "[managed] Training complete after attempt $attempt."
        break
    fi

    echo "[managed] Crashed or incomplete. Restarting in 10s..."
    sleep 10
done
