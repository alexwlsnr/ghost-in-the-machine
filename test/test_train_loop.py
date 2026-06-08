#!/usr/bin/env python3
"""
Tests for the training-loop perf knobs in train_transformer.py:
  - make_batches() — pure batching helper (replaces the hardcoded `range(.., 16)`)
  - --batch-size / batch_size param
  - --qat-every cadence (don't run torch.quantile every step)
  - --amp bf16 autocast path (CUDA-gated)

Local only (imports torch); not part of the CI JS gate.
Run: .venv/bin/python3 test/test_train_loop.py   (or `npm run test:py`)
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
import train_transformer as tt  # noqa: E402


class MakeBatches(unittest.TestCase):
    def test_splits_into_chunks_of_at_most_batch_size(self):
        self.assertEqual(tt.make_batches(list(range(10)), 4),
                         [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])

    def test_covers_every_item_once_in_order(self):
        items = list(range(23))
        flat = [x for b in tt.make_batches(items, 5) for x in b]
        self.assertEqual(flat, items)

    def test_batch_larger_than_input_is_one_batch(self):
        self.assertEqual(tt.make_batches([1, 2, 3], 99), [[1, 2, 3]])

    def test_empty_input(self):
        self.assertEqual(tt.make_batches([], 4), [])

    def test_batch_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            tt.make_batches([1, 2, 3], 0)


def _tiny_model():
    return tt.TinyTransformer(vocab_size=tt.VOCAB_SIZE, d_model=32, n_heads=2,
                              n_layers=1, d_ff=64, max_len=32)


# 8 valid pairs → 8 sequences → 2 batches at batch_size=4
_PAIRS = [
    ("HELLO", "HI THERE FRIEND"), ("HOW ARE YOU", "IM GOOD THANKS"),
    ("BYE", "SEE YOU SOON"), ("THANKS", "NO PROBLEM"),
    ("WHATS UP", "NOT MUCH HERE"), ("GOOD MORNING", "MORNING TO YOU"),
    ("TELL ME A JOKE", "WHY DID IT WORK"), ("WHO ARE YOU", "JUST A GHOST"),
]


class SplitPairs(unittest.TestCase):
    def test_val_frac_zero_keeps_everything_in_train(self):
        pairs = [("a", "b"), ("c", "d"), ("e", "f")]
        train, val = tt.split_pairs(pairs, 0.0)
        self.assertEqual(train, pairs)
        self.assertEqual(val, [])

    def test_splits_disjointly_and_covers_all(self):
        pairs = [(str(i), str(i)) for i in range(8)]
        train, val = tt.split_pairs(pairs, 0.25)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(train), 6)
        self.assertEqual(sorted(train + val), sorted(pairs))  # disjoint union

    def test_deterministic_for_a_given_seed(self):
        pairs = [(str(i), str(i)) for i in range(20)]
        self.assertEqual(tt.split_pairs(pairs, 0.3, seed=7),
                         tt.split_pairs(pairs, 0.3, seed=7))

    def test_always_leaves_at_least_one_training_pair(self):
        train, val = tt.split_pairs([("a", "b"), ("c", "d")], 0.99)
        self.assertGreaterEqual(len(train), 1)

    def test_val_frac_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            tt.split_pairs([("a", "b")], 1.0)


class TrainLoopKnobs(unittest.TestCase):
    def test_qat_every_controls_penalty_cadence(self):
        # 8 seqs / batch 4 = 2 batches/epoch; 2 epochs = 4 steps.
        for qat_every, expected_calls in [(1, 4), (2, 2)]:
            model = _tiny_model()
            n = {"v": 0}
            real = model.compute_quantization_loss

            def counting(weight_bits=4, real=real, n=n):
                n["v"] += 1
                return real(weight_bits)

            model.compute_quantization_loss = counting
            with tempfile.TemporaryDirectory() as d:
                tt.train_transformer(
                    model, _PAIRS, epochs=2, lr=1e-3, device="cpu",
                    checkpoint_file=os.path.join(d, "m.pt"),
                    batch_size=4, qat_every=qat_every,
                )
            self.assertEqual(n["v"], expected_calls, f"qat_every={qat_every}")

    def test_runs_with_custom_batch_size_and_saves_checkpoint(self):
        torch.manual_seed(42)  # fixed seed: ensures acc > 0 so checkpoint is saved
        np.random.seed(42)
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            tt.train_transformer(model, _PAIRS, epochs=5, lr=1e-2, device="cpu",
                                 checkpoint_file=ckpt, batch_size=4)
            self.assertTrue(os.path.exists(ckpt), "checkpoint must be saved when acc improves")
            saved = torch.load(ckpt, weights_only=True, map_location="cpu")
            self.assertTrue(0.0 <= saved["best_acc"] <= 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "amp path needs CUDA")
    def test_amp_bf16_path_runs_on_cuda(self):
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            tt.train_transformer(model, _PAIRS, epochs=2, lr=1e-3, device="cuda",
                                 checkpoint_file=os.path.join(d, "m.pt"),
                                 batch_size=4, amp=True)


class ValidationAndEarlyStop(unittest.TestCase):
    def test_val_split_checkpoints_with_val_loss(self):
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            tt.train_transformer(model, _PAIRS, epochs=3, lr=1e-3, device="cpu",
                                 checkpoint_file=ckpt, batch_size=4, val_frac=0.25)
            self.assertTrue(os.path.exists(ckpt))
            saved = torch.load(ckpt, weights_only=True, map_location="cpu")
            self.assertIsNotNone(saved["best_val_loss"])
            self.assertTrue(math.isfinite(saved["best_val_loss"]))

    def test_early_stops_when_val_stops_improving(self):
        np.random.seed(0)
        torch.manual_seed(0)
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            res = tt.train_transformer(model, _PAIRS, epochs=40, lr=3e-3, device="cpu",
                                       checkpoint_file=os.path.join(d, "m.pt"),
                                       batch_size=4, val_frac=0.25, patience=2)
        self.assertTrue(res["stopped_early"])
        self.assertLess(res["epochs_run"], 40)


class SeparatorToken(unittest.TestCase):
    """Fix A: Q/R separator in make_sequence; Fix B: drop over-length pairs."""

    def test_make_sequence_contains_sep_between_q_and_r(self):
        # The separator must appear exactly once, between Q and R bytes.
        inp, tgt = tt.make_sequence("HI", "HEY", max_len=32)
        q_bytes = list("HI".encode("ascii"))
        r_bytes = list("HEY".encode("ascii"))
        # SEP_TOKEN should appear right after Q bytes
        sep_idx = len(q_bytes)
        self.assertEqual(inp[sep_idx], tt.SEP_TOKEN,
                         f"Expected SEP at idx {sep_idx}, got {inp[sep_idx]}")
        # R bytes follow SEP
        self.assertEqual(inp[sep_idx + 1 : sep_idx + 1 + len(r_bytes)], r_bytes)

    def test_make_sequence_ends_with_eos(self):
        inp, tgt = tt.make_sequence("HI", "HEY", max_len=32)
        self.assertEqual(inp[-1], tt.EOS_TOKEN)

    def test_build_sequences_drops_pairs_that_need_truncation(self):
        # A pair that would overflow max_len should be dropped, not truncated.
        # With SEP + EOS, the threshold is len(q) + 1 + len(r) + 1 <= max_len
        long_q  = "A" * 10
        long_r  = "B" * 10          # 10 + 1 + 10 + 1 = 22, fits max_len=24
        tight_r = "C" * 13          # 10 + 1 + 13 + 1 = 25, doesn't fit max_len=24

        fits_pairs    = [(long_q, long_r)]
        too_long_pairs = [(long_q, tight_r)]

        seqs_ok  = tt._build_sequences(fits_pairs, max_len=24)
        seqs_bad = tt._build_sequences(too_long_pairs, max_len=24)

        self.assertEqual(len(seqs_ok[0]),  1, "Fitting pair should be included")
        self.assertEqual(len(seqs_bad[0]), 0, "Over-length pair should be dropped (truncate=False)")

    def test_build_sequences_truncates_when_flag_set(self):
        # truncate=True: over-length pairs are truncated to fit, not dropped.
        long_q = "A" * 10
        tight_r = "C" * 13  # 10+1+13+1=25, doesn't fit max_len=24 without truncation
        seqs = tt._build_sequences([(long_q, tight_r)], max_len=24, truncate=True)
        self.assertEqual(len(seqs[0]), 1, "Over-length pair should be kept when truncate=True")
        # Sequence must still fit max_len
        self.assertLessEqual(len(seqs[0][0]), 24)

    def test_generate_output_excludes_sep_token(self):
        # generate() should inject SEP after the prompt but strip it from output.
        # Use a stub model that emits SEP then 'X' (88) then EOS.
        class StubSepModel:
            max_len = 32
            scripted = [tt.SEP_TOKEN, 88, tt.EOS_TOKEN]
            calls = 0
            def eval(self): return self
            def to(self, d): return self
            def __call__(self, x):
                tok = self.scripted[self.calls] if self.calls < len(self.scripted) else tt.EOS_TOKEN
                self.calls += 1
                logits = torch.full((1, x.shape[1], tt.VOCAB_SIZE), -10.0)
                logits[0, -1, tok] = 50.0
                return logits

        m = StubSepModel()
        out = tt.generate(m, "HELLO", max_new=10, temperature=1.0, device="cpu")
        self.assertNotIn(chr(tt.SEP_TOKEN), out, "SEP token must not appear in generate() output")
        self.assertEqual(out, "X", f"Expected 'X', got {repr(out)}")


class MultiTurnSequence(unittest.TestCase):
    """Tests for make_sequence_multiturn() — multi-turn training format."""

    def test_two_turn_has_three_seps_and_eos(self):
        # 2 turns → Q1[SEP]R1[SEP]Q2[SEP]R2[EOS] = 3 SEPs (formula: 2N-1)
        turns = [("HELLO", "HEY THERE"), ("HOW ARE YOU", "DOING GREAT")]
        inp, tgt = tt.make_sequence_multiturn(turns, max_len=256)
        self.assertGreater(len(inp), 0)
        self.assertEqual(inp.count(tt.SEP_TOKEN), 3)
        self.assertEqual(inp[-1], tt.EOS_TOKEN)

    def test_single_turn_matches_make_sequence(self):
        # make_sequence_multiturn with one pair should match make_sequence
        turns = [("HELLO", "HEY THERE")]
        inp_mt, tgt_mt = tt.make_sequence_multiturn(turns, max_len=256)
        inp_st, tgt_st = tt.make_sequence("HELLO", "HEY THERE", max_len=256)
        self.assertEqual(inp_mt, inp_st)
        self.assertEqual(tgt_mt, tgt_st)

    def test_target_is_input_shifted_left(self):
        turns = [("HI", "HELLO"), ("HOW ARE YOU", "FINE")]
        inp, tgt = tt.make_sequence_multiturn(turns, max_len=256)
        # tgt[i] == inp[i+1] for all but the last position
        for i in range(len(inp) - 1):
            self.assertEqual(tgt[i], inp[i + 1])

    def test_ends_with_eos_not_sep(self):
        turns = [("HELLO", "HI"), ("BYE", "GOODBYE")]
        inp, _ = tt.make_sequence_multiturn(turns, max_len=256)
        self.assertEqual(inp[-1], tt.EOS_TOKEN)
        self.assertNotEqual(inp[-2], tt.EOS_TOKEN)

    def test_over_length_returns_empty(self):
        turns = [("A" * 60, "B" * 60), ("C" * 60, "D" * 60)]
        inp, tgt = tt.make_sequence_multiturn(turns, max_len=64)
        self.assertEqual(inp, [])
        self.assertEqual(tgt, [])

    def test_layout_q1_sep_r1_sep_q2_sep_r2_eos(self):
        turns = [("HI", "HEY"), ("BYE", "GOODBYE")]
        inp, _ = tt.make_sequence_multiturn(turns, max_len=256)
        # Manually build expected
        q1 = tt.encode("HI")
        r1 = tt.encode("HEY")
        q2 = tt.encode("BYE")
        r2 = tt.encode("GOODBYE")
        expected = q1 + [tt.SEP_TOKEN] + r1 + [tt.SEP_TOKEN] + q2 + [tt.SEP_TOKEN] + r2 + [tt.EOS_TOKEN]
        self.assertEqual(inp, expected)

    def test_parse_multiturn_line_two_turns(self):
        line = "HELLO|HI THERE|HOW ARE YOU|DOING GREAT"
        turns = tt.parse_multiturn_line(line)
        self.assertEqual(turns, [("HELLO", "HI THERE"), ("HOW ARE YOU", "DOING GREAT")])

    def test_parse_multiturn_line_single_turn(self):
        line = "HELLO|HI THERE"
        turns = tt.parse_multiturn_line(line)
        self.assertEqual(turns, [("HELLO", "HI THERE")])

    def test_parse_multiturn_line_odd_fields_drops_last(self):
        # Odd number of pipe-separated fields → drop the last orphan
        line = "A|B|C"
        turns = tt.parse_multiturn_line(line)
        self.assertEqual(turns, [("A", "B")])

    def test_parse_multiturn_line_empty_returns_none(self):
        self.assertIsNone(tt.parse_multiturn_line(""))
        self.assertIsNone(tt.parse_multiturn_line("SINGLE_FIELD_NO_PIPE"))


if __name__ == "__main__":
    unittest.main()
