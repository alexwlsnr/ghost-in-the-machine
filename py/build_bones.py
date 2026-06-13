#!/usr/bin/env python3
"""Bonemaxxing: expand bones.json (187 punchlines for the one true setup) into
training pairs that reinforce the bone-joke signature.

Output: data/bones_train.txt in the standard pipe format (Q|R single-turn,
Q1|R1|Q2|R2 multi-turn). Three forms per punchline, for variety:
  1. joke-request  -> full joke (setup + punchline)        [single-turn]
  2. the setup     -> punchline                            [single-turn]
  3. joke-request -> setup ; "why?" -> punchline           [two-turn delivery]

Blend it in deliberately (it's a SIGNATURE, kept on purpose — see CLAUDE.md), but
don't let it dominate: a few hundred pairs against a ~2.5M-line corpus is a small
fraction. Tiny models mode-collapse onto bone jokes if over-weighted, so cap the
dose (1× is plenty; the v2 corpus already had ~13% skeleton-themed joke replies).

    .venv/bin/python3 py/build_bones.py            # -> data/bones_train.txt
"""
import json, os, random

SETUP = "Why did the bones go to the party?"
REQUESTS = [
    "Tell me a joke.", "Tell me a joke about bones.", "Got any jokes?",
    "Make me laugh.", "Tell me something funny.", "Know any good jokes?",
    "Say something funny.", "Can you tell a joke?", "Cheer me up.",
    "Do you know any jokes?",
]
FOLLOWUPS = ["Why?", "Why's that?", "Go on.", "And?", "Because?"]

def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jokes = json.load(open(os.path.join(here, "bones.json")))
    rng = random.Random(42)
    pairs = []
    for j in jokes:
        setup, punch = j["setup"], j["punchline"]
        full = f"{setup.upper()} {punch}"
        # 1. request -> full joke (two distinct request phrasings per punchline)
        for req in rng.sample(REQUESTS, 2):
            pairs.append(f"{req}|{full}")
        # 2. setup -> punchline (direct)
        pairs.append(f"{setup}|{punch}")
        # 3. two-turn suspense delivery
        req = rng.choice(REQUESTS)
        fu = rng.choice(FOLLOWUPS)
        pairs.append(f"{req}|{setup.upper()}|{fu}|{punch}")

    # de-dup, stable order
    seen, out = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p); out.append(p)
    dest = os.path.join(here, "data", "bones_train.txt")
    open(dest, "w").write("\n".join(out) + "\n")
    print(f"wrote {dest} — {len(out)} pairs from {len(jokes)} punchlines")

if __name__ == "__main__":
    main()
