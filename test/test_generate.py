#!/usr/bin/env python3
"""
Tests for train_transformer.generate() prompt handling (fixes doc #6).

The bug: `tokens = tokens[:model.max_len - max_new]` truncated the prompt (with
the defaults, "HELLO" → "HELL"), and `prompt_len` was recomputed from the full
prompt, dropping the first generated token in the returned slice.

Uses a deterministic stub model — no torch weights, no sampling randomness — so
the assertions are exact. Local only (imports torch); not part of the CI JS gate.

Run: .venv/bin/python3 test/test_generate.py   (or `npm run test:py`)
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
from train_transformer import generate, encode, EOS_TOKEN, VOCAB_SIZE  # noqa: E402


class StubModel:
    """Records the sequence length fed on each step; emits a scripted token list,
    then EOS. Forces a token by spiking its logit."""

    def __init__(self, max_len, scripted):
        self.max_len = max_len
        self.scripted = list(scripted)
        self.calls = 0
        self.seen_lens = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        self.seen_lens.append(x.shape[1])
        tok = self.scripted[self.calls] if self.calls < len(self.scripted) else EOS_TOKEN
        self.calls += 1
        logits = torch.full((1, x.shape[1], VOCAB_SIZE), -10.0)
        logits[0, -1, tok] = 50.0
        return logits


class GeneratePromptHandling(unittest.TestCase):
    def test_feeds_full_prompt_when_it_fits(self):
        # "HELLO" (5 bytes) fits in ctx 64 — the model must see all 5 tokens,
        # not max_len - max_new = 4.
        m = StubModel(max_len=64, scripted=[])  # emit EOS immediately
        out = generate(m, "HELLO", max_new=60, temperature=1.0, device="cpu")
        self.assertEqual(m.seen_lens[0], len(encode("HELLO")))
        self.assertEqual(out, "")

    def test_does_not_drop_first_generated_token(self):
        # Emit one real token ('X' = 88) then EOS; it must appear in the output.
        m = StubModel(max_len=64, scripted=[88])
        out = generate(m, "HELLO", max_new=60, temperature=1.0, device="cpu")
        self.assertEqual(out, "X")

    def test_caps_prompt_at_context_window(self):
        m = StubModel(max_len=64, scripted=[])
        generate(m, "A" * 100, max_new=60, temperature=1.0, device="cpu")
        self.assertEqual(m.seen_lens[0], 63)  # max_len - 1


if __name__ == "__main__":
    unittest.main()
