#!/usr/bin/env bash
# GPU power/performance sweep benchmark.
# Tests inference tok/s at each power limit and finds the efficiency sweet spot.
#
# Usage (run from repo root):
#   bash scripts/bench_power_sweep.sh 5080    # RTX 5080 sweep
#   bash scripts/bench_power_sweep.sh 3070    # RTX 3070 sweep
#   bash scripts/bench_power_sweep.sh custom 150 200 250 300  # custom wattages
#
# Requires: sudo nvidia-smi -pl <watts> permission
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

GPU="${1:-5080}"

# Training benchmark arch — defaults to Ternary Shade target (40M params)
# Override with e.g. ARCH=classic D_MODEL=256 N_LAYERS=4 D_FF=1024
ARCH="${ARCH:-ternary}"
D_MODEL="${D_MODEL:-512}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-6}"
D_FF="${D_FF:-2048}"
MAX_LEN="${MAX_LEN:-256}"
BATCH="${BATCH:-32}"

case "$GPU" in
  5080)   LIMITS="150 200 250 300 360" ;;
  3070)   LIMITS="120 140 165 185 220" ;;
  custom) shift; LIMITS="$*" ;;
  *)      echo "Usage: $0 [5080|3070|custom <watts...>]"; exit 1 ;;
esac

# Check sudo for nvidia-smi
if ! sudo nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
  echo "ERROR: sudo nvidia-smi required to set power limits"
  echo "Run: sudo visudo  and add:"
  echo "  $(whoami) ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi"
  exit 1
fi

echo "Training Throughput Sweep — $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Arch: $ARCH  d=$D_MODEL  L=$N_LAYERS  ff=$D_FF  batch=$BATCH  ctx=$MAX_LEN"
echo "Limits to test: $LIMITS W"
echo ""

# Get TDP to restore at the end
ORIGINAL_LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | xargs printf "%.0f")

cleanup() {
  echo ""
  echo "Restoring original power limit: ${ORIGINAL_LIMIT}W"
  sudo nvidia-smi -pl "$ORIGINAL_LIMIT" >/dev/null
}
trap cleanup EXIT

for WATTS in $LIMITS; do
  echo "─── Testing ${WATTS}W ───"
  sudo nvidia-smi -pl "$WATTS" >/dev/null
  sleep 2  # let power limit stabilise
  .venv/bin/python3 py/bench_training.py \
    --arch "$ARCH" --d-model "$D_MODEL" --n-heads "$N_HEADS" \
    --n-layers "$N_LAYERS" --d-ff "$D_FF" --max-len "$MAX_LEN" \
    --batch-size "$BATCH" --amp
  echo ""
done

echo "═══ Sweep complete. Results in logs/gpu_power_bench.json ═══"
.venv/bin/python3 -c "
import json
results = json.load(open('logs/gpu_power_bench.json'))
gpu = results[-1]['gpu']
arch = results[-1].get('arch', 'classic')
rows = [r for r in results if r['gpu']==gpu and r.get('mode')=='training' and r.get('arch')==arch]
rows = rows[-5:]  # last sweep only
rows.sort(key=lambda r: r['power_limit_w'])
if not rows: exit()
best_tps = max(r['tok_per_sec'] for r in rows)
best_eff = max(r['tok_per_watt'] for r in rows)
print(f'\n{gpu}  |  arch={arch}  d={rows[0][\"d_model\"]}  L={rows[0][\"n_layers\"]}  batch={rows[0][\"batch_size\"]}')
print(f'{\"Limit\":>7}  {\"Draw\":>6}  {\"Ktok/s\":>8}  {\"tok/s/W\":>8}  {\"rel %\":>6}')
print(chr(9472) * 52)
for r in rows:
    flags = []
    if r['tok_per_sec'] == best_tps: flags.append('⚡ fastest')
    if r['tok_per_watt'] == best_eff: flags.append('🌿 most efficient')
    print(f'  {r[\"power_limit_w\"]:>5.0f}W  {r[\"avg_power_w\"]:>5.0f}W  {r[\"tok_per_sec\"]/1000:>7.1f}K  {r[\"tok_per_watt\"]:>8.0f}  {int(r[\"tok_per_sec\"]/best_tps*100):>5}%  {\" \".join(flags)}')
"
