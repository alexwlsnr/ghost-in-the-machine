#!/usr/bin/env python3
"""
training_status.py — lightweight status emitter for the training supervision harness.

The trainer calls write_status() after each epoch; the watch_training.sh watcher
polls the resulting status.json to decide when to wake Claude.

Usage as a library:
    from training_status import write_status, read_status
    write_status("logs/shade_status.json", epoch=42, val_loss=1.189, state="running")

Usage as a CLI (initial status write from a launch script):
    python3 py/training_status.py --write logs/shade_status.json \\
        --tier shade --phase training --state running \\
        --epochs-total 30 --checkpoint ckpt/shade_fp32.pt
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone


# ─── Core helpers ──────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_status(path: str) -> dict | None:
    """Read and parse a status.json file. Returns None if missing or corrupt."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_status(path: str, **fields) -> None:
    """Merge fields into the existing status file (or create from scratch).

    Writes atomically: data goes to <path>.tmp then os.rename() to <path>.
    Always updates `updated_at` to now.

    Args:
        path:    Destination JSON file path.
        **fields: Any subset of the status schema fields.
    """
    # Load existing data so we can merge (preserves fields not in this update)
    existing = read_status(path) or {}

    # Merge new fields over existing
    existing.update(fields)

    # Always refresh updated_at
    existing["updated_at"] = _now_iso()

    # Set started_at on first write if not already present
    if "started_at" not in existing:
        existing["started_at"] = existing["updated_at"]

    # Set pid from current process if not supplied
    if "pid" not in existing:
        existing["pid"] = os.getpid()

    # Atomic write: write to .tmp then rename
    tmp_path = path + ".tmp"
    # Ensure parent directory exists
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    with open(tmp_path, "w") as f:
        json.dump(existing, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.rename(tmp_path, path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Write or read a training status.json file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--write", metavar="PATH",
                   help="Write/merge status fields to this path and exit.")
    p.add_argument("--read", metavar="PATH",
                   help="Print the current status JSON and exit.")

    # Status fields (all optional; whatever is supplied gets merged)
    p.add_argument("--tier", help="Model tier name (wisp/shade/specter)")
    p.add_argument("--phase", default="training", help="Phase label (default: training)")
    p.add_argument("--state",
                   choices=["running", "done", "failed", "early_stopped"],
                   help="Current training state")
    p.add_argument("--epoch", type=int, help="Current epoch number")
    p.add_argument("--epochs-total", type=int, dest="epochs_total",
                   help="Total planned epochs")
    p.add_argument("--train-loss", type=float, dest="train_loss")
    p.add_argument("--val-loss", type=float, dest="val_loss")
    p.add_argument("--best-val-loss", type=float, dest="best_val_loss")
    p.add_argument("--best-epoch", type=int, dest="best_epoch")
    p.add_argument("--stopped-early", action="store_true", dest="stopped_early")
    p.add_argument("--checkpoint", help="Path to current checkpoint file")
    p.add_argument("--eta-seconds", type=int, dest="eta_seconds",
                   help="Estimated seconds remaining")
    p.add_argument("--pid", type=int, help="Training process PID")
    return p


def main(argv=None):
    p = _build_parser()
    args = p.parse_args(argv)

    if args.read:
        status = read_status(args.read)
        if status is None:
            print(f"No status found at {args.read}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(status, indent=2))
        return

    if args.write:
        # Collect only the fields that were explicitly provided
        fields = {}
        for key in ("tier", "phase", "state", "epoch", "epochs_total",
                    "train_loss", "val_loss", "best_val_loss", "best_epoch",
                    "checkpoint", "eta_seconds", "pid"):
            val = getattr(args, key, None)
            if val is not None:
                fields[key] = val
        if args.stopped_early:
            fields["stopped_early"] = True
        write_status(args.write, **fields)
        return

    p.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
