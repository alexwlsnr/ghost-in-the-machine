#!/usr/bin/env bash
# Chain: wait for quality → run scale → run domain → done
set -euo pipefail

VENV=".venv/bin/python3"
BASE_CKPT="ckpt/shade_bpe_ternary.pt"
COMMON_ARGS="--tokenizer data/bpe_4096.json --arch ternary --d-model 512 --n-heads 8 --n-layers 4 --d-ff 2048 --max-len 256 --batch-size 32 --lr 0.0002 --epochs 50 --val-frac 0.02 --patience 10 --truncate --amp --preserve-case --mask-query-loss --device cuda"

is_done() {
  local f="$1"
  local attempt state
  for attempt in 1 2 3; do
    state=$(python3 -c "import json,sys; d=json.load(open('$f')); print(d.get('state',''))" 2>/dev/null) && break
    sleep 2
  done
  [[ "$state" == "done" || "$state" == "early_stopped" ]]
}

run_ablation() {
  local name="$1" file="$2" ckpt="$3" status="$4" log="$5"
  echo "$(date -u +%H:%M:%S) Starting $name ablation..."
  $VENV -u py/train_transformer.py \
    --resume "$BASE_CKPT" \
    --checkpoint "$ckpt" \
    --status-file "$status" \
    --file "$file" \
    $COMMON_ARGS 2>&1 | tee "$log"
  echo "$(date -u +%H:%M:%S) $name ablation finished."
}

# ── Wait for quality to finish ──────────────────────────────────────────────
echo "Waiting for quality ablation to finish..."
while ! is_done logs/shade_bpe_quality_status.json; do
  sleep 30
done
echo "Quality done. Starting scale..."

# ── Scale ablation ──────────────────────────────────────────────────────────
run_ablation "scale" data/ablation_scale.txt \
  ckpt/shade_bpe_scale.pt logs/shade_bpe_scale_status.json logs/shade_bpe_scale_train.log

# ── Domain ablation ─────────────────────────────────────────────────────────
echo "Starting domain ablation..."
run_ablation "domain" data/ablation_domain.txt \
  ckpt/shade_bpe_domain.pt logs/shade_bpe_domain_status.json logs/shade_bpe_domain_train.log

echo "$(date -u +%H:%M:%S) All ablations complete. Next: Shade BPE Ternary Modern."
