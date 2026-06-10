#!/usr/bin/env bash
# Chained training manager:
#   1. Shade BPE Ternary — resumes from existing checkpoint, auto-restarts on CUDA crash
#   2. Wisp BPE Ternary  — fresh retrain (old corrupted checkpoint discarded), auto-restarts
#
# Run in a tmux session:
#   tmux new -s train_chain
#   bash scripts/train_chain_shade_then_wisp_bpe.sh

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV=".venv/bin/python3"

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[chain $(date -u '+%H:%M:%S UTC')] $*"; }

is_done() {
    local status_file="$1"
    local attempt
    for attempt in 1 2 3; do
        result=$($VENV -c "
import json, sys
try:
    d = json.load(open('$status_file'))
    done = d.get('state') in ('done','early_stopped') or bool(d.get('stopped_early'))
    sys.exit(0 if done else 1)
except Exception as e:
    sys.exit(2)
" 2>/dev/null; echo $?)
        [ "$result" = "0" ] && return 0
        [ "$result" = "1" ] && return 1
        # exit code 2 = parse error — wait and retry
        sleep 2
    done
    return 1
}

run_managed() {
    local label="$1"; shift
    local status_file="$1"; shift
    local ckpt_file="$1"; shift
    # remaining args passed to train_transformer.py

    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        log "[$label] attempt $attempt"

        if is_done "$status_file"; then
            log "[$label] already done — skipping"
            return 0
        fi

        local resume_flag=""
        if [ -f "$ckpt_file" ]; then
            resume_flag="--resume $ckpt_file"
        fi

        $VENV -u py/train_transformer.py \
            $resume_flag \
            --checkpoint "$ckpt_file" \
            --status-file "$status_file" \
            "$@" || true   # don't abort chain on crash

        sleep 3  # let status file flush before checking
        if is_done "$status_file"; then
            log "[$label] training complete after $attempt attempt(s)"
            return 0
        fi

        log "[$label] crashed or incomplete — restarting in 15s…"
        sleep 15
    done
}

# ── 1. Shade BPE Ternary ─────────────────────────────────────────────────────

log "=== STAGE 1: Shade BPE Ternary ==="
run_managed "shade_bpe_ternary" \
    "logs/shade_bpe_ternary_status.json" \
    "ckpt/shade_bpe_ternary.pt" \
    --file      data/shade_bpe_train.txt \
    --tokenizer data/bpe_4096.json \
    --arch      ternary \
    --d-model   512 \
    --n-heads   8 \
    --n-layers  4 \
    --d-ff      2048 \
    --max-len   256 \
    --batch-size 32 \
    --lr        0.0008 \
    --epochs    100 \
    --val-frac  0.02 \
    --patience  20 \
    --truncate  \
    --amp       \
    --preserve-case \
    --mask-query-loss \
    --device    cuda

log "=== STAGE 1 complete ==="

# ── 2. Wisp BPE Ternary (fresh) ──────────────────────────────────────────────

log "=== STAGE 2: Wisp BPE Ternary (fresh retrain) ==="

# The old checkpoint was corrupted by the best_val_loss reset bug (epoch 1, loss 2.057).
# Discard it so training starts clean.
if [ -f "ckpt/wisp_bpe_ternary.pt" ]; then
    mv "ckpt/wisp_bpe_ternary.pt" "ckpt/wisp_bpe_ternary.pt.corrupted"
    log "Renamed corrupted Wisp checkpoint → wisp_bpe_ternary.pt.corrupted"
fi

run_managed "wisp_bpe_ternary" \
    "logs/wisp_bpe_ternary_status.json" \
    "ckpt/wisp_bpe_ternary.pt" \
    --file      data/bpe_ternary_train.txt \
    --tokenizer data/bpe_4096.json \
    --arch      ternary \
    --d-model   320 \
    --n-heads   4 \
    --n-layers  4 \
    --d-ff      1280 \
    --max-len   256 \
    --batch-size 16 \
    --lr        0.001 \
    --epochs    150 \
    --val-frac  0.02 \
    --patience  15 \
    --truncate  \
    --amp       \
    --mask-query-loss \
    --device    cuda

log "=== STAGE 2 complete ==="
log "=== Chain finished. Both models trained. ==="
