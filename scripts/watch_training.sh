#!/usr/bin/env bash
# watch_training.sh — poll a training status.json and exit when something noteworthy happens.
#
# Exits (waking the caller / Claude) on:
#   - State transitions to: done | failed | early_stopped
#   - Stale heartbeat: updated_at is >5 min old AND state is "running" (hung job)
#   - Periodic check-in: every 45 minutes (prints heartbeat, exits)
#
# On exit prints a one-line summary:
#   [EVENT] <state> | epoch <n>/<total> | val_loss=<x> | eta=<Xs>
#
# Usage:
#   bash scripts/watch_training.sh [STATUS_JSON_PATH]
#
# Default path: status.json in the current working directory.
#
# Requirements: bash, date, grep, sed — no python, no jq.

set -euo pipefail

STATUS_FILE="${1:-status.json}"
POLL_INTERVAL=60          # seconds between polls
STALE_THRESHOLD=300       # 5 minutes: mark job as hung if no update
CHECKIN_INTERVAL=2700     # 45 minutes: periodic wake even if all quiet

started_at_mono=$(date +%s)
last_checkin_mono=$started_at_mono

# ─── JSON field extraction (no jq) ────────────────────────────────────────────
# Extracts the value of a top-level string or number field from a simple JSON file.
# Works for fields produced by json.dump(indent=2) — one key per line.

_field() {
    local key="$1" file="$2"
    # Match: "key": "value"  or  "key": value (number/bool/null)
    grep -m1 "\"${key}\":" "$file" 2>/dev/null \
      | sed 's/.*"'"${key}"'":[[:space:]]*//' \
      | sed 's/,[[:space:]]*$//' \
      | sed 's/^"\(.*\)"$/\1/'
}

# ─── Parse updated_at → Unix epoch (portable, no GNU date extensions assumed) ─
# Format: 2026-06-07T21:00:00Z
_iso_to_epoch() {
    local ts="$1"
    # Extract components
    local year month day hour min sec
    year="${ts:0:4}"
    month="${ts:5:2}"
    day="${ts:8:2}"
    hour="${ts:11:2}"
    min="${ts:14:2}"
    sec="${ts:17:2}"
    # Use date -d (GNU) or date -j (BSD). Try GNU first.
    if date --version >/dev/null 2>&1; then
        date -d "${year}-${month}-${day}T${hour}:${min}:${sec}Z" +%s 2>/dev/null \
            || echo 0
    else
        # BSD / macOS
        date -j -f "%Y-%m-%dT%H:%M:%SZ" "${ts}" +%s 2>/dev/null \
            || echo 0
    fi
}

# ─── Emit the one-line exit summary ───────────────────────────────────────────
_emit() {
    local event="$1" file="$2"
    local state epoch epochs_total val_loss eta

    if [[ -f "$file" ]]; then
        state=$(_field "state" "$file")
        epoch=$(_field "epoch" "$file")
        epochs_total=$(_field "epochs_total" "$file")
        val_loss=$(_field "val_loss" "$file")
        eta=$(_field "eta_seconds" "$file")
    else
        state="unknown"
        epoch="-"
        epochs_total="-"
        val_loss="-"
        eta="-"
    fi

    # Defaults for missing/null fields
    [[ -z "$state"        || "$state"        == "null" ]] && state="unknown"
    [[ -z "$epoch"        || "$epoch"        == "null" ]] && epoch="-"
    [[ -z "$epochs_total" || "$epochs_total" == "null" ]] && epochs_total="-"
    [[ -z "$val_loss"     || "$val_loss"     == "null" ]] && val_loss="-"
    [[ -z "$eta"          || "$eta"          == "null" ]] && eta="-"

    echo "[${event}] ${state} | epoch ${epoch}/${epochs_total} | val_loss=${val_loss} | eta=${eta}s"
}

# ─── Main loop ────────────────────────────────────────────────────────────────

echo "[watch_training] polling ${STATUS_FILE} every ${POLL_INTERVAL}s (stale=${STALE_THRESHOLD}s, checkin=${CHECKIN_INTERVAL}s)"

while true; do
    now_mono=$(date +%s)

    # ── Periodic check-in (every 45 min, regardless of state) ──────────────
    elapsed_since_checkin=$(( now_mono - last_checkin_mono ))
    if (( elapsed_since_checkin >= CHECKIN_INTERVAL )); then
        last_checkin_mono=$now_mono
        _emit "HEARTBEAT" "$STATUS_FILE"
        exit 0
    fi

    # ── Read the status file ────────────────────────────────────────────────
    if [[ ! -f "$STATUS_FILE" ]]; then
        echo "[watch_training] status file not found yet: ${STATUS_FILE}" >&2
        sleep "$POLL_INTERVAL"
        continue
    fi

    state=$(_field "state" "$STATUS_FILE")
    updated_at=$(_field "updated_at" "$STATUS_FILE")

    # ── Terminal state check ────────────────────────────────────────────────
    case "$state" in
        done|failed|early_stopped)
            _emit "EVENT" "$STATUS_FILE"
            exit 0
            ;;
    esac

    # ── Stale heartbeat check (only while state=running) ───────────────────
    if [[ "$state" == "running" && -n "$updated_at" && "$updated_at" != "null" ]]; then
        updated_epoch=$(_iso_to_epoch "$updated_at")
        if [[ "$updated_epoch" -gt 0 ]]; then
            age=$(( now_mono - updated_epoch ))
            if (( age >= STALE_THRESHOLD )); then
                echo "[watch_training] stale: last update ${age}s ago (threshold ${STALE_THRESHOLD}s)" >&2
                _emit "STALE" "$STATUS_FILE"
                exit 0
            fi
        fi
    fi

    sleep "$POLL_INTERVAL"
done
