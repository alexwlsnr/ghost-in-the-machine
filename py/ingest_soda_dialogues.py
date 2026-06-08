#!/usr/bin/env python3
"""
Extract full multi-turn dialogues from SODA for Spec512 training.

Unlike ingest_soda.py (which flattens conversations into Q|R pairs),
this preserves the full turn structure as Q1|R1|Q2|R2|... per line.
Spec512 at ctx=1024 can fit 5-8 turns per sequence — enough for real
multi-turn context learning.

Usage:
  python3 py/ingest_soda_dialogues.py --output data/soda_dialogues.txt
  python3 py/ingest_soda_dialogues.py --output data/soda_dialogues.txt --max-turns 6
"""

import argparse
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_soda import normalise_names

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_TURN_LEN = 3
MAX_TURN_LEN = 120
DEFAULT_CTX   = 1024
DEFAULT_MAX_TURNS = 8


# ── Pure functions (all tested) ───────────────────────────────────────────────

def normalise_turn(turn: str) -> str:
    """Strip, uppercase, collapse whitespace, strip quotes, normalise names."""
    t = turn.strip().strip('"\'').strip()
    t = re.sub(r'\s+', ' ', t).upper()
    t = normalise_names(t)
    return t


def filter_turn(turn: str) -> bool:
    """Return True if the turn is usable as a training token."""
    if not (MIN_TURN_LEN <= len(turn) <= MAX_TURN_LEN):
        return False
    try:
        turn.encode('ascii')
    except UnicodeEncodeError:
        return False
    if not all(32 <= ord(c) <= 126 for c in turn):
        return False
    if '|' in turn:
        return False
    return True


def extract_dialogue(
    raw_turns: list[str],
    max_ctx: int = DEFAULT_CTX,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> Optional[list[str]]:
    """
    Normalise and filter a list of raw dialogue turns.

    Returns a list of even-length normalised turns that fit within max_ctx,
    or None if the result is too short to be useful.
    Trims from the front when the dialogue exceeds the context budget.
    """
    if len(raw_turns) < 2:
        return None

    # Normalise and filter each turn
    normed = [normalise_turn(t) for t in raw_turns]
    normed = [t for t in normed if filter_turn(t)]

    # Must have at least one complete Q/R pair
    if len(normed) < 2:
        return None

    # Enforce even length (complete Q/R pairs only)
    if len(normed) % 2 != 0:
        normed = normed[:-1]

    # Cap at max_turns
    if len(normed) > max_turns:
        normed = normed[-max_turns:]   # keep the most recent turns
        if len(normed) % 2 != 0:
            normed = normed[1:]

    # Trim from the front until sequence fits in ctx budget
    # Budget: sum of byte lengths + one SEP per turn + one EOS
    def seq_len(turns: list[str]) -> int:
        return sum(len(t) for t in turns) + len(turns) + 1

    while len(normed) >= 2 and seq_len(normed) > max_ctx:
        normed = normed[2:]   # drop oldest Q/R pair

    if len(normed) < 2:
        return None

    return normed


def format_dialogue(turns: list[str]) -> str:
    """Serialise a list of turns as a pipe-separated line."""
    return '|'.join(turns)


# ── Ingestion pipeline ────────────────────────────────────────────────────────

def ingest(
    output_path: str,
    max_ctx: int = DEFAULT_CTX,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_dialogues: Optional[int] = None,
    split: str = "train",
    verbose: bool = True,
) -> int:
    """Stream SODA, extract full dialogues, write Q1|R1|Q2|R2|... lines."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Loading allenai/soda ({split} split) — streaming…")

    ds = load_dataset("allenai/soda", split=split, streaming=True)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    n_written = 0
    n_seen = 0
    n_skipped = 0

    with open(output_path, "w") as out:
        for example in ds:
            raw_turns = example.get("dialogue", [])
            n_seen += 1

            turns = extract_dialogue(raw_turns, max_ctx=max_ctx, max_turns=max_turns)
            if turns is None:
                n_skipped += 1
                continue

            out.write(format_dialogue(turns) + "\n")
            n_written += 1

            if n_written % 10_000 == 0 and verbose:
                pct_skip = 100 * n_skipped / max(n_seen, 1)
                print(f"  {n_written:,} dialogues  ({n_seen:,} seen, {pct_skip:.0f}% skipped)")

            if max_dialogues and n_written >= max_dialogues:
                break

    if verbose:
        print(f"\nDone: {n_written:,} dialogues → {output_path}")
        print(f"      {n_seen:,} seen, {n_skipped:,} skipped ({100*n_skipped/max(n_seen,1):.0f}%)")

    return n_written


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract multi-turn SODA dialogues")
    parser.add_argument("--output",    "-o", default="data/soda_dialogues.txt")
    parser.add_argument("--max-ctx",   type=int, default=DEFAULT_CTX)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-dialogues", "-n", type=int, default=None)
    parser.add_argument("--split",     default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--quiet",     "-q", action="store_true")
    args = parser.parse_args()

    ingest(
        output_path=args.output,
        max_ctx=args.max_ctx,
        max_turns=args.max_turns,
        max_dialogues=args.max_dialogues,
        split=args.split,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
