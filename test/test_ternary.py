#!/usr/bin/env python3
"""
TDD tests for the ternary architecture.

Tests TernaryLinear (STE training), TinyTransformerTernary, and the
ternary serialization format. Run before any production code is written.

Run: .venv/bin/python3 test/test_ternary.py
"""

import sys, os, math, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))

import unittest
import torch
import torch.nn.functional as F
import numpy as np


class TestTernaryLinear(unittest.TestCase):

    def test_gradients_flow_via_ste(self):
        """STE: weight.grad must be non-zero after backward through ternary forward."""
        from train_transformer import TernaryLinear
        layer = TernaryLinear(8, 4)
        x = torch.randn(2, 8)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertFalse(torch.all(layer.weight.grad == 0),
                         "All gradients are zero — STE is broken")

    def test_forward_output_shape(self):
        """Output shape must be (batch, out_features)."""
        from train_transformer import TernaryLinear
        layer = TernaryLinear(16, 8)
        x = torch.randn(3, 16)
        out = layer(x)
        self.assertEqual(out.shape, (3, 8))

    def test_effective_weights_are_ternary(self):
        """Weights applied in forward pass must only be {-scale, 0, +scale}."""
        from train_transformer import TernaryLinear
        torch.manual_seed(42)
        layer = TernaryLinear(8, 4, bias=False)
        # Use identity input to expose each weight column individually
        x = torch.eye(8)           # (8, 8)
        with torch.no_grad():
            out = layer(x)         # (8, 4) — rows are weight columns
        effective_w = out.T        # (4, 8) — the applied weight matrix
        scale = layer.weight.abs().mean().item()
        for val in effective_w.flatten().tolist():
            is_zero  = abs(val) < 1e-5
            is_scale = abs(abs(val) - scale) < 1e-5
            self.assertTrue(is_zero or is_scale,
                f"Effective weight {val:.6f} is not ternary (scale={scale:.6f})")

    def test_roughly_half_weights_are_zero(self):
        """With absmean threshold, ~50% of weights should collapse to zero."""
        from train_transformer import TernaryLinear
        torch.manual_seed(0)
        layer = TernaryLinear(256, 256, bias=False)
        # Standard init is approximately normal — absmean threshold ≈ 0.5*mean(|w|)
        # gives roughly 40-60% zero sparsity
        scale = layer.weight.abs().mean()
        threshold = 0.5 * scale
        zero_frac = (layer.weight.abs() < threshold).float().mean().item()
        self.assertGreater(zero_frac, 0.30, "Fewer than 30% zero weights — threshold too low")
        self.assertLess(zero_frac, 0.70, "More than 70% zero weights — threshold too high")


class TestTinyTransformerTernary(unittest.TestCase):

    def _make_model(self):
        from train_transformer import TinyTransformerTernary
        return TinyTransformerTernary(
            vocab_size=258, d_model=64, n_heads=4,
            n_layers=2, d_ff=128, max_len=64,
        )

    def test_forward_output_shape(self):
        """Forward must produce (B, T, vocab_size) logits."""
        model = self._make_model()
        x = torch.randint(0, 258, (2, 16))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 16, 258))

    def test_loss_decreases_over_training_steps(self):
        """Loss must decrease over 5 gradient steps — verifies convergence signal."""
        model = self._make_model()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, 258, (4, 16))
        y = torch.randint(0, 258, (4, 16))

        losses = []
        for _ in range(5):
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 258), y.reshape(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0],
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}")

    def test_quantization_loss_returns_zero(self):
        """Ternary model does not need extra QAT penalty — must return 0."""
        model = self._make_model()
        qat_loss = model.compute_quantization_loss()
        self.assertEqual(qat_loss.item(), 0.0)

    def test_arch_attribute_is_ternary(self):
        """Model must identify itself as ternary for serializer routing."""
        model = self._make_model()
        self.assertEqual(model.arch, 'ternary')

    def test_attention_is_causal(self):
        """Future tokens must not influence past token logits."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            x = torch.randint(0, 258, (1, 8))
            full_out  = model(x)
            short_out = model(x[:, :4])
        # First 4 token logits must match whether or not future tokens are present
        self.assertTrue(
            torch.allclose(full_out[:, :4], short_out, atol=1e-4),
            "Causality violated: future tokens leak into past positions"
        )


class TestTernaryQuantization(unittest.TestCase):

    def test_quantize_ternary_output_size(self):
        """Packed output must be ceil(n/4) bytes."""
        from serialize import quantize_ternary
        for n in [4, 8, 15, 16, 17, 100]:
            t = torch.randn(n)
            packed, scale = quantize_ternary(t)
            expected_bytes = math.ceil(n / 4)
            self.assertEqual(len(packed), expected_bytes,
                f"n={n}: expected {expected_bytes} bytes, got {len(packed)}")

    def test_quantize_ternary_scale_is_absmean(self):
        """Scale must equal the absmean of the tensor."""
        from serialize import quantize_ternary
        t = torch.tensor([1.0, -2.0, 0.5, -0.5])
        _, scale = quantize_ternary(t)
        expected = t.abs().mean().item()
        self.assertAlmostEqual(scale, expected, places=5)

    def test_quantize_ternary_roundtrip(self):
        """Unpacked values must all be in {-1, 0, +1} (before scale)."""
        from serialize import quantize_ternary
        torch.manual_seed(7)
        t = torch.randn(32)
        packed, scale = quantize_ternary(t)
        # Unpack
        unpacked = []
        for byte in packed:
            for shift in (6, 4, 2, 0):
                code = (byte >> shift) & 0x3
                if code == 0: unpacked.append(-1)
                elif code == 1: unpacked.append(0)
                elif code == 2: unpacked.append(1)
        unpacked = unpacked[:len(t)]
        unique = set(unpacked)
        self.assertTrue(unique.issubset({-1, 0, 1}),
            f"Got unexpected ternary codes: {unique - {-1, 0, 1}}")

    def test_positive_values_above_threshold_become_plus_one(self):
        """Values >= 0.5 * absmean must map to code +1."""
        from serialize import quantize_ternary
        # All weights = 1.0: absmean=1.0, threshold=0.5, all → +1 (code=2)
        t = torch.ones(4)
        packed, scale = quantize_ternary(t)
        byte = packed[0]
        for shift in (6, 4, 2, 0):
            code = (byte >> shift) & 0x3
            self.assertEqual(code, 2, f"Expected code 2 (+1), got {code}")

    def test_near_zero_values_become_zero(self):
        """Values < 0.5 * absmean must map to code 0 (zero)."""
        from serialize import quantize_ternary
        # Mix: large values push threshold high, small values collapse to 0
        t = torch.tensor([10.0, -10.0, 0.01, -0.01])
        packed, scale = quantize_ternary(t)
        unpacked_codes = []
        for byte in packed:
            for shift in (6, 4, 2, 0):
                unpacked_codes.append((byte >> shift) & 0x3)
        unpacked_codes = unpacked_codes[:4]
        # Indices 2 and 3 (0.01, -0.01) should be code 1 (zero)
        self.assertEqual(unpacked_codes[2], 1,
            f"Small positive should be zero (code=1), got {unpacked_codes[2]}")
        self.assertEqual(unpacked_codes[3], 1,
            f"Small negative should be zero (code=1), got {unpacked_codes[3]}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
