#!/usr/bin/env python3
"""
Build the Spec512 v1.2 clean training dataset.

Combines:
  - GHOST/HUMAN scenario dialogues (all turns: 1-turn, 2-turn, 3-turn)
  - Filtered SODA dialogues (emotional/reactions/small_talk strata only)
  - Meta/jokes distilled pairs

Output: data/spec512_v12_clean.txt

Usage:
  python3 py/build_v12_dataset.py
  python3 py/build_v12_dataset.py --soda-limit 80000 --output data/spec512_v12_clean.txt
"""

import argparse
import random
import re
from pathlib import Path

# ── Strata keyword sets for SODA filtering ────────────────────────────────────

_EMOTIONAL_KW = {
    'SAD', 'ANXIOUS', 'DEPRESS', 'LONELY', 'FEAR', 'HURT', 'UPSET',
    'EXCIT', 'HAPPY', 'JOY', 'LOVE', 'ANGER', 'FRUSTRAT', 'WORRIED',
    'DISAPPOINT', 'PROUD', 'GRIEF', 'MOURN', 'MISS YOU', 'NERVOUS',
    'SORRY TO HEAR', 'THAT MUST BE', 'I UNDERSTAND', 'HOW ARE YOU FEELING',
    'FEEL SO', 'FEELING SO', 'I FEEL', 'YOU FEEL',
}
_REACTION_KW = {
    'WOW', 'AMAZING', 'REALLY?', 'OH NO', 'CONGRAT', 'GREAT NEWS',
    'AWESOME', 'FANTASTIC', 'UNBELIEVABLE', 'I KNOW RIGHT', "CAN'T BELIEVE",
    'NO WAY', 'ISN\'T IT', 'THAT\'S SO', "YOU'RE KIDDING",
}
_SMALL_TALK_KW = {
    'HOW ARE', 'WHAT ARE YOU UP TO', 'HOW IS', 'HOW HAVE YOU',
    'HOW DO YOU', 'WHAT HAVE YOU', 'LONG DAY', 'SOUNDS LIKE',
    'JUST CHATTING', 'JUST TALKING', 'NOTHING MUCH', 'SAME OLD',
    'WHAT\'S NEW', 'WHAT\'S UP', 'NOT MUCH', 'CHILLING',
}

# Profanity / aggression filter (remove dialogues containing these)
_BLOCKLIST = {
    'FUCKING', 'FUCK YOU', 'ASSHOLE', 'BULLSHIT', 'BITCH',
    'YOU IDIOT', 'HATE YOU', 'KILL YOURSELF',
}

# Strata that signal purely transactional / out-of-distribution content
_TRANSACTIONAL_KW = {
    'PER DAY', 'INVOICE', 'TOTAL COST', 'HOW MUCH DO YOU CHARGE',
    'CAPITAL OF', 'WHAT IS THE CAPITAL', 'INTEREST RATE',
    'FILE YOUR TAXES', 'WRITE A REPORT',
}


def _matches_any(text: str, kws: set) -> bool:
    return any(k in text for k in kws)


def is_soda_valid(line: str) -> bool:
    """Return True if the SODA dialogue line passes quality filters."""
    u = line.upper()

    # Must be normalised (HUMAN present) — checks both sides using the
    # pipe-separated format written by ingest_soda_dialogues.py
    if 'HUMAN' not in u:
        return False

    # Reject transactional / factual / out-of-distribution
    if _matches_any(u, _TRANSACTIONAL_KW):
        return False

    # Reject aggressive/profane content
    if _matches_any(u, _BLOCKLIST):
        return False

    # Accept emotional, reactions, or small_talk stratum
    return (
        _matches_any(u, _EMOTIONAL_KW)
        or _matches_any(u, _REACTION_KW)
        or _matches_any(u, _SMALL_TALK_KW)
    )


def load_lines(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        print(f"  [skip] {path} not found")
        return []
    lines = [l.rstrip('\n') for l in p.open() if l.strip()]
    print(f"  Loaded {len(lines):,} lines from {path}")
    return lines


def deduplicate(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/spec512_v12_clean.txt')
    parser.add_argument('--soda-source', default='data/soda_dialogues.txt')
    parser.add_argument('--soda-limit', type=int, default=80_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_lines: list[str] = []

    print("=== Building Spec512 v1.2 dataset ===\n")

    # ── 1. Scenario dialogues ──────────────────────────────────────────────────
    print("Scenarios:")
    scenario_sources = [
        'data/scenarios_multiturn.txt',  # 5K existing 2-4 turn
        'data/scenarios.txt',            # 5K single-turn
        'data/scenarios_3turn.txt',      # new 20K 3-turn
        'data/scenarios_2turn.txt',      # new 20K 2-turn
    ]
    scenario_lines = []
    for src in scenario_sources:
        scenario_lines.extend(load_lines(src))
    scenario_lines = deduplicate(scenario_lines)
    print(f"  Scenario total (deduped): {len(scenario_lines):,}")
    all_lines.extend(scenario_lines)

    # ── 2. Filtered SODA dialogues ─────────────────────────────────────────────
    print(f"\nSODA (filtering {args.soda_source}):")
    soda_raw = load_lines(args.soda_source)
    rng.shuffle(soda_raw)
    soda_filtered = []
    rejected = 0
    for line in soda_raw:
        if is_soda_valid(line):
            soda_filtered.append(line)
            if len(soda_filtered) >= args.soda_limit:
                break
        else:
            rejected += 1
    print(f"  Accepted: {len(soda_filtered):,}  Rejected: {rejected:,}")
    all_lines.extend(soda_filtered)

    # ── 3. Meta / jokes distilled pairs ───────────────────────────────────────
    print("\nMeta/jokes:")
    meta_sources = ['data/meta.txt', 'data/jokes.txt', 'data/greetings.txt']
    meta_lines = []
    for src in meta_sources:
        meta_lines.extend(load_lines(src))
    meta_lines = deduplicate(meta_lines)
    print(f"  Meta/jokes total (deduped): {len(meta_lines):,}")
    # Repeat to ~5K so model sees them enough
    target_meta = 5_000
    repeated: list[str] = []
    while len(repeated) < target_meta:
        repeated.extend(meta_lines)
    repeated = repeated[:target_meta]
    all_lines.extend(repeated)

    # ── 4. Shuffle and write ───────────────────────────────────────────────────
    rng.shuffle(all_lines)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(all_lines) + '\n')

    # Summary
    single = sum(1 for l in all_lines if l.count('|') == 1)
    two_turn = sum(1 for l in all_lines if l.count('|') == 3)
    three_plus = sum(1 for l in all_lines if l.count('|') >= 5)
    print(f"\n=== Dataset written → {out} ===")
    print(f"  Total lines:    {len(all_lines):,}")
    print(f"  Single-turn:    {single:,}")
    print(f"  2-turn:         {two_turn:,}")
    print(f"  3+ turn:        {three_plus:,}")


if __name__ == '__main__':
    main()
