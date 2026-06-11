#!/usr/bin/env bash
# Wait for scenario generation jobs to complete, then build dataset and train.
# Run from repo root: bash scripts/run_v12_pipeline.sh
set -euo pipefail

TARGET_3TURN=20000
TARGET_2TURN=20000
DATASET="data/spec512_v12_clean.txt"

echo "[$(date '+%H:%M:%S')] Waiting for generation jobs..."

while true; do
    lines_3=$(wc -l < data/scenarios_3turn.txt 2>/dev/null || echo 0)
    lines_2=$(wc -l < data/scenarios_2turn.txt 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] 3-turn: $lines_3/$TARGET_3TURN  2-turn: $lines_2/$TARGET_2TURN"

    if [ "$lines_3" -ge "$TARGET_3TURN" ] && [ "$lines_2" -ge "$TARGET_2TURN" ]; then
        echo "[$(date '+%H:%M:%S')] Both generation jobs complete!"
        break
    fi
    sleep 60
done

echo ""
echo "[$(date '+%H:%M:%S')] Building dataset..."
.venv/bin/python3 py/build_v12_dataset.py --output "$DATASET"

echo ""
echo "[$(date '+%H:%M:%S')] Launching training..."
bash scripts/train_spec512_v12.sh

echo "[$(date '+%H:%M:%S')] Training launched. Monitor: tail -f logs/spec512_v12_train.log"
