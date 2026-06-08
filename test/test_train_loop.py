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
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "m.pt")
            tt.train_transformer(model, _PAIRS, epochs=2, lr=1e-3, device="cpu",
                                 checkpoint_file=ckpt, batch_size=4)
            self.assertTrue(os.path.exists(ckpt))
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


if __name__ == "__main__":
    unittest.main()
