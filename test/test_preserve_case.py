#!/usr/bin/env python3
"""
test_preserve_case.py — Tests for the --preserve-case flag in train_transformer.py

The flag is the critical blocker for Wraith: without it, technical content like
`-la`, `chmod 755`, `find . -name` gets uppercased and the model learns garbage.

Run: python3 test/test_preserve_case.py   (or npm run test:py)
"""

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
import train_transformer as tt  # noqa: E402


# ─── make_sequence (case-agnostic — caller decides) ─────────────────────────

class TestMakeSequence(unittest.TestCase):
    """make_sequence itself never uppercases — the caller (train_transformer) decides."""

    def test_preserves_lowercase_command(self):
        """make_sequence must not uppercase its inputs."""
        inp, tgt = tt.make_sequence("ls -la", "list files", 32)
        decoded = tt.decode(inp)
        self.assertIn("ls -la", decoded,
                      "make_sequence must not uppercase — it should receive pre-processed text")

    def test_preserves_mixed_case(self):
        """Flags like -a, -l, --help must survive make_sequence unchanged."""
        inp, tgt = tt.make_sequence("find . -name '*.py'", "find python files", 64)
        decoded = tt.decode(inp)
        self.assertIn("-name", decoded)

    def test_uppercase_caller_still_works(self):
        """Old callers that uppercase before calling should still work fine."""
        inp, tgt = tt.make_sequence("HELLO".upper(), "HI THERE".upper(), 32)
        decoded = tt.decode(inp)
        self.assertIn("HELLO", decoded)


# ─── train_transformer preserve_case param ──────────────────────────────────

def _tiny_model():
    return tt.TinyTransformer(
        vocab_size=tt.VOCAB_SIZE,
        d_model=32, n_heads=2, n_layers=1, d_ff=64, max_len=64,
    )


# Pairs that contain case-sensitive technical content
_LINUX_PAIRS = [
    ("what does ls -la do?", "list all files in long format including hidden files"),
    ("how do I find files by name?", "find . -name 'filename'"),
    ("what does chmod 755 do?", "sets owner rwx, group and others r-x"),
    ("how to show disk usage?", "du -sh directory"),
    ("grep recursive search", "grep -r 'pattern' /path/to/dir"),
    ("how do I list hidden files?", "ls -a"),
    ("what is tar -xzf?", "extract a gzip-compressed tar archive"),
    ("show running processes", "ps aux"),
]


class TestTrainTransformerPreserveCase(unittest.TestCase):

    def test_preserve_case_false_uppercases_training_data(self):
        """Default behavior (preserve_case=False) must uppercase all pairs."""
        recorded_sequences = []
        real_make_sequence = tt.make_sequence

        def spy(*args, **kwargs):
            recorded_sequences.append((args[0], args[1]))
            return real_make_sequence(*args, **kwargs)

        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            # Monkeypatch make_sequence to record inputs
            tt.make_sequence = spy
            try:
                tt.train_transformer(
                    model, _LINUX_PAIRS[:4], epochs=1, lr=1e-3,
                    device="cpu",
                    checkpoint_file=os.path.join(d, "m.pt"),
                    batch_size=4,
                    preserve_case=False,
                )
            finally:
                tt.make_sequence = real_make_sequence

        # Every query and response should be uppercase
        for q, r in recorded_sequences:
            self.assertEqual(q, q.upper(),
                             f"Expected uppercase query, got: {q!r}")
            self.assertEqual(r, r.upper(),
                             f"Expected uppercase response, got: {r!r}")

    def test_preserve_case_true_keeps_original_case(self):
        """With preserve_case=True, queries and responses must NOT be uppercased."""
        recorded_sequences = []
        real_make_sequence = tt.make_sequence

        def spy(*args, **kwargs):
            recorded_sequences.append((args[0], args[1]))
            return real_make_sequence(*args, **kwargs)

        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            tt.make_sequence = spy
            try:
                tt.train_transformer(
                    model, _LINUX_PAIRS[:4], epochs=1, lr=1e-3,
                    device="cpu",
                    checkpoint_file=os.path.join(d, "m.pt"),
                    batch_size=4,
                    preserve_case=True,
                )
            finally:
                tt.make_sequence = real_make_sequence

        # At least some sequences should contain lowercase content
        has_lowercase = any(
            q != q.upper() or r != r.upper()
            for q, r in recorded_sequences
        )
        self.assertTrue(has_lowercase,
                        "preserve_case=True must keep original case in training sequences")

    def test_preserve_case_keeps_flags_intact(self):
        """Flags like -la, -r, --name must survive training data preparation."""
        recorded_sequences = []
        real_make_sequence = tt.make_sequence

        def spy(*args, **kwargs):
            recorded_sequences.append((args[0], args[1]))
            return real_make_sequence(*args, **kwargs)

        # Use only pairs with flags in the response
        flag_pairs = [
            ("list hidden files", "ls -a"),
            ("long format listing", "ls -la"),
            ("grep recursive", "grep -r 'pattern' dir"),
        ]

        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            tt.make_sequence = spy
            try:
                tt.train_transformer(
                    model, flag_pairs, epochs=1, lr=1e-3,
                    device="cpu",
                    checkpoint_file=os.path.join(d, "m.pt"),
                    batch_size=4,
                    preserve_case=True,
                )
            finally:
                tt.make_sequence = real_make_sequence

        responses = [r for _, r in recorded_sequences]
        self.assertTrue(any("-a" in r or "-la" in r or "-r" in r for r in responses),
                        "Flags like -la, -r must be preserved when preserve_case=True. "
                        f"Got responses: {responses}")

    def test_default_is_uppercase_backward_compat(self):
        """Omitting preserve_case must default to the old uppercasing behavior."""
        recorded_sequences = []
        real_make_sequence = tt.make_sequence

        def spy(*args, **kwargs):
            recorded_sequences.append((args[0], args[1]))
            return real_make_sequence(*args, **kwargs)

        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            tt.make_sequence = spy
            try:
                # Call without preserve_case — should default to uppercase
                tt.train_transformer(
                    model, _LINUX_PAIRS[:2], epochs=1, lr=1e-3,
                    device="cpu",
                    checkpoint_file=os.path.join(d, "m.pt"),
                    batch_size=4,
                    # no preserve_case kwarg
                )
            finally:
                tt.make_sequence = real_make_sequence

        for q, r in recorded_sequences:
            self.assertEqual(q, q.upper(), f"Default must uppercase: {q!r}")


# ─── generate() preserve_case param ─────────────────────────────────────────

class TestGeneratePreservesCase(unittest.TestCase):
    """generate() with preserve_case=True must NOT uppercase the prompt."""

    def test_generate_uppercases_by_default(self):
        """Without preserve_case, generate() encodes the uppercased prompt."""
        model = _tiny_model()
        # We can't easily intercept encode(), so we verify via the token stream:
        # "ls -la" uppercased is "LS -LA"; check the prompt tokens match uppercase.
        prompt = "ls -la"
        # encode is deterministic
        upper_tokens = tt.encode(prompt.upper())
        normal_tokens = tt.encode(prompt)
        self.assertNotEqual(upper_tokens, normal_tokens,
                             "These must differ to make the test meaningful")

        # generate() with default (preserve_case=False) should encode prompt.upper()
        # We monkeypatch encode to capture what it receives
        captured = []
        real_encode = tt.encode

        def spy_encode(text):
            captured.append(text)
            return real_encode(text)

        tt.encode = spy_encode
        try:
            tt.generate(model, prompt, max_new=5, temperature=1.0, device="cpu",
                        preserve_case=False)
        finally:
            tt.encode = real_encode

        # The first captured encode call is the prompt
        self.assertTrue(len(captured) >= 1)
        self.assertEqual(captured[0], prompt.upper(),
                         f"Expected encode('{prompt.upper()}'), got encode({captured[0]!r})")

    def test_generate_preserves_case_when_flag_set(self):
        """With preserve_case=True, generate() must NOT uppercase the prompt."""
        model = _tiny_model()
        prompt = "ls -la"

        captured = []
        real_encode = tt.encode

        def spy_encode(text):
            captured.append(text)
            return real_encode(text)

        tt.encode = spy_encode
        try:
            tt.generate(model, prompt, max_new=5, temperature=1.0, device="cpu",
                        preserve_case=True)
        finally:
            tt.encode = real_encode

        self.assertTrue(len(captured) >= 1)
        self.assertEqual(captured[0], prompt,
                         f"preserve_case=True must pass prompt as-is to encode. "
                         f"Expected {prompt!r}, got {captured[0]!r}")

    def test_generate_default_is_uppercase(self):
        """generate() called without preserve_case must uppercase (backward compat)."""
        model = _tiny_model()
        prompt = "grep pattern"

        captured = []
        real_encode = tt.encode

        def spy_encode(text):
            captured.append(text)
            return real_encode(text)

        tt.encode = spy_encode
        try:
            tt.generate(model, prompt, max_new=5, temperature=1.0, device="cpu")
        finally:
            tt.encode = real_encode

        self.assertTrue(len(captured) >= 1)
        self.assertEqual(captured[0], prompt.upper(),
                         "Default must uppercase for backward compat")


if __name__ == "__main__":
    unittest.main(verbosity=2)
