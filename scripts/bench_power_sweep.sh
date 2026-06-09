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
MODEL="${MODEL:-ckpt/spec512_v2.pt}"

case "$GPU" in
  5080)   LIMITS="150 200 250 300 360" ;;
  3070)   LIMITS="120 140 165 185 220" ;;
  custom) shift; LIMITS="$*" ;;
  *)      echo "Usage: $0 [5080|3070|custom <watts...>]"; exit 1 ;;
esac

if [ ! -f "$MODEL" ]; then
  echo "Model not found: $MODEL"
  echo "Set MODEL=ckpt/yourmodel.pt to override"
  exit 1
fi

# Check sudo for nvidia-smi
if ! sudo nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
  echo "ERROR: sudo nvidia-smi required to set power limits"
  echo "Run: sudo visudo  and add:"
  echo "  $(whoami) ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi"
  exit 1
fi

echo "Power/Performance Sweep — $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Model: $MODEL"
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
  .venv/bin/python3 py/bench_gpu_power.py --model "$MODEL"
  echo ""
done

echo "═══ Sweep complete. Results in logs/gpu_power_bench.json ═══"
.venv/bin/python3 -c "
import json, sys
results = json.load(open('logs/gpu_power_bench.json'))
# Only show this GPU's results for this model
import os
model = os.path.basename('$MODEL')
gpu = results[-1]['gpu']
rows = [r for r in results if r['gpu']==gpu and r['model']==model]
rows.sort(key=lambda r: r['power_limit_w'])
if not rows: sys.exit()
best_tps = max(r['tok_per_sec'] for r in rows)
best_eff = max(r['tok_per_watt'] for r in rows)
print(f'\n{gpu}  |  {model}')
print(f'{\"Limit\":>7}  {\"Draw\":>6}  {\"tok/s\":>7}  {\"tok/s/W\":>8}  {\"rel %\":>6}')
print('─' * 50)
for r in rows:
    flags = []
    if r['tok_per_sec']  == best_tps: flags.append('⚡ best speed')
    if r['tok_per_watt'] == best_eff: flags.append('🌿 best efficiency')
    print(f'  {r[\"power_limit_w\"]:>5.0f}W  {r[\"avg_power_w\"]:>5.0f}W  {r[\"tok_per_sec\"]:>7.1f}  {r[\"tok_per_watt\"]:>8.3f}  {int(r[\"tok_per_sec\"]/best_tps*100):>5}%  {\" \".join(flags)}')
"
