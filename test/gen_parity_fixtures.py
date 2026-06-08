#!/usr/bin/env python3
"""
Generate PyTorch reference logits for the end-to-end parity test.

The TS/Wasm forward pass is checked against these. PyTorch is the source of
truth: if the serializer or TS orchestrator rework changes the logits beyond
tolerance, test/parity.test.js fails.

Run from repo root:
  .venv/bin/python3 test/gen_parity_fixtures.py
Writes test/fixtures/parity_logits.json (commit it).
"""

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))
from train_transformer import TinyTransformer, encode  # noqa: E402

CHECKPOINT = os.path.join(ROOT, "ckpt", "wisp_ctx128.pt")  # ctx=128 model
OUT = os.path.join(ROOT, "test", "fixtures", "parity_logits.json")

# Fixed prompts spanning a few lengths. Kept short so the sequence stays well
# inside max_len. Tokens are stored explicitly so both sides decode identically.
PROMPTS = ["HI", "HELLO", "HOW ARE YOU", "TELL ME A JOKE", "GOODBYE"]


def main():
    ckpt = torch.load(CHECKPOINT, weights_only=True, map_location="cpu")
    arch = ckpt["architecture"]
    model = TinyTransformer(
        vocab_size=arch["vocab_size"], d_model=arch["d_model"],
        n_heads=arch["n_heads"], n_layers=arch["n_layers"],
        d_ff=arch["d_ff"], max_len=arch["max_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cases = []
    with torch.no_grad():
        for prompt in PROMPTS:
            tokens = encode(prompt.upper())
            logits = model(torch.tensor([tokens]))  # (1, seq, vocab)
            last = logits[0, -1].tolist()
            cases.append({
                "prompt": prompt,
                "tokens": tokens,
                "argmax_last": int(logits[0, -1].argmax()),
                "logits_last": last,
            })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "checkpoint": os.path.basename(CHECKPOINT),
            "architecture": arch,
            "cases": cases,
        }, f)
    print(f"Wrote {len(cases)} cases → {os.path.relpath(OUT, ROOT)}")
    for c in cases:
        print(f"  {c['prompt']:16s} seq={len(c['tokens'])} argmax_last={c['argmax_last']}")


if __name__ == "__main__":
    main()
