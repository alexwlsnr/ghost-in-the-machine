#!/usr/bin/env python3
"""
Ingest the SODA social dialogue dataset (allenai/soda, CC-BY 4.0) into our
Q|R training format.

SODA contains ~1.2M multi-turn social conversations. Each consecutive pair
of dialogue turns becomes a Q|R training pair. After ASCII/length/ctx
filtering we expect ~1.5-2M usable pairs — enough for all three model tiers.

Usage:
  python3 py/ingest_soda.py --output data/soda_pairs.txt --max-pairs 100000
  python3 py/ingest_soda.py --output data/soda_all.txt   # full dataset
"""

import argparse
import os
import re
import sys
from typing import Optional


# ── Filter thresholds ─────────────────────────────────────────────────────────

MIN_Q_LEN = 5    # bytes
MAX_Q_LEN = 90
MIN_R_LEN = 5    # bytes
MAX_R_LEN = 120
DEFAULT_CTX = 128


# ── Pure functions (all tested) ───────────────────────────────────────────────

def normalise_pair(query: str, response: str) -> tuple[str, str]:
    """Strip, collapse whitespace, remove surrounding quotes, uppercase."""
    def norm(s: str) -> str:
        s = s.strip().strip('"\'').strip()
        s = re.sub(r'\s+', ' ', s)
        return s.upper()
    return norm(query), norm(response)


def filter_pair(query: str, response: str, max_ctx: int = DEFAULT_CTX) -> bool:
    """Return True if the pair passes all quality filters."""
    if not query or not response:
        return False
    # ASCII-only (byte-level model requires printable ASCII)
    try:
        query.encode('ascii')
        response.encode('ascii')
    except UnicodeEncodeError:
        return False
    # Must be printable ASCII only
    if not all(32 <= ord(c) <= 126 for c in query + response):
        return False
    # Length bounds
    if not (MIN_Q_LEN <= len(query) <= MAX_Q_LEN):
        return False
    if not (MIN_R_LEN <= len(response) <= MAX_R_LEN):
        return False
    # Context window: Q + SEP + R + EOS must fit
    if len(query) + 1 + len(response) + 1 > max_ctx:
        return False
    # No pipe character (would corrupt our Q|R format)
    if '|' in query or '|' in response:
        return False
    return True


def extract_pairs(dialogue: list[str]) -> list[tuple[str, str]]:
    """Extract consecutive turn pairs from a dialogue list.

    Each adjacent (turn[i], turn[i+1]) becomes a (query, response) pair,
    giving N-1 pairs from an N-turn conversation.
    Drops turns containing the pipe character.
    """
    pairs = []
    for i in range(len(dialogue) - 1):
        q = dialogue[i]
        r = dialogue[i + 1]
        if '|' not in q and '|' not in r:
            pairs.append((q, r))
    return pairs


def dedup_pairs(
    pairs: list[tuple[str, str]],
    by_query: bool = False,
) -> list[tuple[str, str]]:
    """Deduplicate pairs. by_query=True keeps only the first pair per query."""
    seen: set = set()
    result = []
    for q, r in pairs:
        key = q if by_query else (q, r)
        if key not in seen:
            seen.add(key)
            result.append((q, r))
    return result


# ── Ingestion pipeline ────────────────────────────────────────────────────────

def ingest(
    output_path: str,
    max_pairs: Optional[int] = None,
    max_ctx: int = DEFAULT_CTX,
    split: str = "train",
    dedup_by_query: bool = True,
    verbose: bool = True,
) -> int:
    """Stream SODA, filter, and write Q|R pairs. Returns number written."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Loading allenai/soda ({split} split) — streaming…")

    ds = load_dataset("allenai/soda", split=split, streaming=True, trust_remote_code=True)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    seen_queries: set[str] = set()
    n_written = 0
    n_seen = 0
    n_filtered = 0

    with open(output_path, "w") as out:
        for example in ds:
            dialogue = example.get("dialogue", [])
            raw_pairs = extract_pairs(dialogue)

            for raw_q, raw_r in raw_pairs:
                n_seen += 1
                q, r = normalise_pair(raw_q, raw_r)

                if not filter_pair(q, r, max_ctx):
                    n_filtered += 1
                    continue

                if dedup_by_query and q in seen_queries:
                    n_filtered += 1
                    continue

                seen_queries.add(q)
                out.write(f"{q}|{r}\n")
                n_written += 1

                if n_written % 50_000 == 0 and verbose:
                    pct = 100 * n_filtered / max(n_seen, 1)
                    print(f"  {n_written:,} written  ({n_seen:,} seen, {pct:.0f}% filtered)")

                if max_pairs and n_written >= max_pairs:
                    break

            if max_pairs and n_written >= max_pairs:
                break

    if verbose:
        pct = 100 * n_filtered / max(n_seen, 1)
        print(f"\nDone: {n_written:,} pairs → {output_path}")
        print(f"      {n_seen:,} total seen, {pct:.0f}% filtered")

    return n_written


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest SODA social dialogue dataset")
    parser.add_argument("--output", "-o", default="data/soda_pairs.txt")
    parser.add_argument("--max-pairs", "-n", type=int, default=None,
                        help="Stop after N pairs (omit for full dataset ~1.5M)")
    parser.add_argument("--max-ctx", type=int, default=DEFAULT_CTX,
                        help=f"Context window budget for filtering (default {DEFAULT_CTX})")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--no-dedup-query", action="store_true",
                        help="Keep multiple responses per query (more pairs, less diverse)")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    n = ingest(
        output_path=args.output,
        max_pairs=args.max_pairs,
        max_ctx=args.max_ctx,
        split=args.split,
        dedup_by_query=not args.no_dedup_query,
        verbose=not args.quiet,
    )
    print(f"\n✓ {n:,} pairs written to {args.output}")
    print(f"\nNext: python3 py/train_transformer.py --file {args.output} ...")


if __name__ == "__main__":
    main()
