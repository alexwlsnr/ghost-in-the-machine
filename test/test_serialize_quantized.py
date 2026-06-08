#!/usr/bin/env python3
"""
Quantized serialization tests.

Builds tiny synthetic models (no checkpoint file needed), serializes them
in fp32, 8-bit, and 4-bit modes, then exercises the serializer logic directly
to verify:

  1. Manifest structure is correct for each format.
  2. Binary sizes match expected compression ratios.
  3. fp32 round-trip is near-identical (< 1e-4 max abs).
  4. 8-bit quantization noise is within tolerance (< 0.05 max abs on scale-1 weights).
  5. 4-bit quantization noise is within tolerance (< 0.5 max abs on scale-1 weights).
  6. Quantization of zeros stays zero.
  7. Per-tensor scales are stored in sections (not a separate top-level dict).

Run:
  .venv/bin/python3 test/test_serialize_quantized.py
or:
  python3 test/test_serialize_quantized.py  (if torch/numpy available)
"""

import sys
import json
import math
import tempfile
import struct
from pathlib import Path

import numpy as np
import torch

# Add project root to path so serialize.py is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "py"))

from serialize import quantize_4bit, quantize_4bit_grouped, quantize_8bit, serialize


# ─── Tiny model factory ─────────────────────────────────────────────

def make_tiny_checkpoint(
    vocab_size: int = 16,
    d_model: int = 8,
    n_heads: int = 2,
    n_layers: int = 1,
    d_ff: int = 16,
    max_len: int = 8,
    seed: int = 42,
) -> dict:
    """Build a minimal random checkpoint dict matching the model schema."""
    torch.manual_seed(seed)
    state = {}

    state["token_embed.weight"] = torch.randn(vocab_size, d_model) * 0.1
    state["pos_embed.weight"] = torch.randn(max_len, d_model) * 0.1

    for li in range(n_layers):
        ls = f"encoder.layers.{li}"
        state[f"{ls}.norm1.weight"] = torch.ones(d_model)
        state[f"{ls}.norm1.bias"] = torch.zeros(d_model)
        state[f"{ls}.norm2.weight"] = torch.ones(d_model)
        state[f"{ls}.norm2.bias"] = torch.zeros(d_model)
        # in_proj_weight / bias: shape (3*d, d)
        state[f"{ls}.self_attn.in_proj_weight"] = torch.randn(3 * d_model, d_model) * 0.3
        state[f"{ls}.self_attn.in_proj_bias"] = torch.zeros(3 * d_model)
        state[f"{ls}.self_attn.out_proj.weight"] = torch.randn(d_model, d_model) * 0.3
        state[f"{ls}.self_attn.out_proj.bias"] = torch.zeros(d_model)
        state[f"{ls}.linear1.weight"] = torch.randn(d_ff, d_model) * 0.3
        state[f"{ls}.linear1.bias"] = torch.zeros(d_ff)
        state[f"{ls}.linear2.weight"] = torch.randn(d_model, d_ff) * 0.3
        state[f"{ls}.linear2.bias"] = torch.zeros(d_model)

    state["ln_final.weight"] = torch.ones(d_model)
    state["ln_final.bias"] = torch.zeros(d_model)
    state["head.weight"] = torch.randn(vocab_size, d_model) * 0.1

    arch = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "max_len": max_len,
    }
    return {"model_state": state, "architecture": arch}


# ─── Helper: serialize to a temp dir ────────────────────────────────

def serialize_tiny(weight_bits: int, seed: int = 42) -> tuple[dict, bytes]:
    """Serialize a tiny synthetic model; return (manifest, binary_data)."""
    ck = make_tiny_checkpoint(seed=seed)

    # Patch torch.load to return our fake checkpoint
    import unittest.mock as mock
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = str(Path(tmpdir) / "model")
        # Write a dummy .pt (we'll patch torch.load)
        dummy_pt = prefix + ".pt"
        Path(dummy_pt).write_bytes(b"dummy")

        with mock.patch("serialize.torch.load", return_value=ck):
            serialize(dummy_pt, prefix, weight_bits)

        manifest = json.loads(Path(prefix + ".json").read_text())
        binary = Path(prefix + ".bin").read_bytes()

    return manifest, binary


# ─── Tests ──────────────────────────────────────────────────────────

PASS = 0
FAIL = 0


def check(condition: bool, msg: str) -> None:
    global PASS, FAIL
    if condition:
        print(f"  PASS  {msg}")
        PASS += 1
    else:
        print(f"  FAIL  {msg}")
        FAIL += 1


def test_manifest_structure() -> None:
    print("\n=== Manifest structure ===")
    for bits in [32, 8, 4]:
        manifest, _ = serialize_tiny(bits)

        check("version" in manifest, f"bits={bits}: manifest has 'version'")
        check("weight_format" in manifest, f"bits={bits}: manifest has 'weight_format'")
        check("sections" in manifest, f"bits={bits}: manifest has 'sections'")
        check("architecture" in manifest, f"bits={bits}: manifest has 'architecture'")
        # Old top-level 'scales' key should NOT be present (scales are per-section now)
        check("scales" not in manifest, f"bits={bits}: no top-level 'scales' key")

        expected_fmt = {32: "fp32", 8: "int8", 4: "int4g"}[bits]
        check(manifest["weight_format"] == expected_fmt,
              f"bits={bits}: weight_format == '{expected_fmt}'")

        # Check that quantized sections have appropriate scale fields
        for name, sec in manifest["sections"].items():
            dtype = sec["dtype"]
            if dtype == 'int8':
                check("scale" in sec, f"bits={bits}: section '{name}' has scale")
            elif dtype == 'int4g':
                check("scales_offset" in sec, f"bits={bits}: section '{name}' has scales_offset")
            else:
                check(dtype == "float32", f"bits={bits}: section '{name}' dtype is float32")
                check("scale" not in sec, f"bits={bits}: fp32 section '{name}' has no scale")


def test_compression_ratios() -> None:
    print("\n=== Compression ratios ===")
    # Count only the quantizable sections (attention/FFN weights)
    # Embeddings, head, norms, biases are always fp32

    manifest32, bin32 = serialize_tiny(32)
    manifest8, bin8 = serialize_tiny(8)
    manifest4, bin4 = serialize_tiny(4)

    size32 = len(bin32)
    size8 = len(bin8)
    size4 = len(bin4)

    # 8-bit should be smaller than fp32 (quantized weight sections are 4x smaller)
    check(size8 < size32, f"8-bit binary ({size8}B) < fp32 ({size32}B)")
    # 4-bit should be smaller than 8-bit
    check(size4 < size8, f"4-bit binary ({size4}B) < 8-bit ({size8}B)")

    print(f"    fp32={size32}B  8bit={size8}B  4bit={size4}B")


def test_fp32_roundtrip() -> None:
    print("\n=== fp32 round-trip (< 1e-4) ===")
    # Verify that fp32 sections round-trip exactly and quantized sections stay fp32
    manifest, binary = serialize_tiny(32)

    for name, sec in manifest["sections"].items():
        check(sec["dtype"] == "float32", f"fp32 mode: '{name}' dtype == float32")

    # Spot-check: lnf_w should be all-ones (from make_tiny_checkpoint)
    sec = manifest["sections"]["lnf_w"]
    data = np.frombuffer(binary[sec["offset"]:sec["offset"] + sec["size"]], dtype=np.float32)
    diff = np.max(np.abs(data - 1.0))
    check(diff < 1e-4, f"lnf_w round-trip error {diff:.2e} < 1e-4")


def test_8bit_quantization_accuracy() -> None:
    print("\n=== 8-bit quantization accuracy ===")
    # Test quantize_8bit directly: round-trip error should be < max_abs / 127
    torch.manual_seed(1)
    for shape in [(8, 8), (32, 16), (64, 8)]:
        t = torch.randn(*shape) * 0.5
        raw, scale = quantize_8bit(t)
        q = np.frombuffer(raw, dtype=np.int8).reshape(shape)
        reconstructed = q.astype(np.float32) * scale
        original = t.numpy()
        max_abs_err = np.max(np.abs(reconstructed - original))
        expected_max = scale  # quantization error ≤ 0.5 * scale ≤ max_abs / 127
        check(max_abs_err < expected_max + 1e-6,
              f"8bit shape={shape}: max_abs_err {max_abs_err:.4f} < {expected_max:.4f}")

    # Manifest-level: quantized weights have 'scale' field
    manifest, _ = serialize_tiny(8)
    for name, sec in manifest["sections"].items():
        if sec["dtype"] == "int8":
            check("scale" in sec and sec["scale"] > 0,
                  f"8bit section '{name}' has positive scale")


def test_4bit_quantization_accuracy() -> None:
    print("\n=== 4-bit quantization accuracy ===")
    # Test quantize_4bit directly: max quantization error ≤ scale (= absmax/7)
    torch.manual_seed(2)
    for shape in [(8, 8), (16, 8)]:
        t = torch.randn(*shape) * 0.5
        raw, scale = quantize_4bit(t)
        # Unpack: high nibble = even index, low nibble = odd index
        flat = t.numpy().ravel()
        data = np.frombuffer(raw, dtype=np.uint8)
        reconstructed = np.zeros(len(flat), dtype=np.float32)
        for i in range(len(flat)):
            byte = int(data[i // 2])
            nibble = (byte >> 4) & 0xF if i % 2 == 0 else byte & 0xF
            reconstructed[i] = (nibble - 8) * scale
        max_abs_err = np.max(np.abs(reconstructed - flat))
        expected_max = scale + 1e-5  # quantization error ≤ 0.5 * scale + rounding
        check(max_abs_err < expected_max + 1e-4,
              f"4bit shape={shape}: max_abs_err {max_abs_err:.4f} < {expected_max:.4f}")

    # Manifest-level: quantized weights have 'scale' field
    manifest, _ = serialize_tiny(4)
    for name, sec in manifest["sections"].items():
        if sec["dtype"] == "int4":
            check("scale" in sec and sec["scale"] > 0,
                  f"4bit section '{name}' has positive scale")


def test_zeros_stay_zero() -> None:
    print("\n=== Zero tensors round-trip as zero ===")
    # quantize_4bit and quantize_8bit should handle all-zero tensors gracefully
    z = torch.zeros(8, 8)

    raw4, scale4 = quantize_4bit(z)
    data4 = np.frombuffer(raw4, dtype=np.uint8)
    # All bytes should be 0x88 (nibbles 8 and 8 = value 0 in offset-binary)
    expected_byte = (8 << 4) | 8
    check(all(b == expected_byte for b in data4),
          f"4bit zeros: all bytes == 0x{expected_byte:02X} (got {set(data4)})")

    raw8, scale8 = quantize_8bit(z)
    data8 = np.frombuffer(raw8, dtype=np.int8)
    check(np.all(data8 == 0), "8bit zeros: all bytes == 0")


def test_mixed_precision_layout() -> None:
    print("\n=== Mixed-precision layout: fp32-only sections ===")
    # In 8-bit and 4-bit modes, certain sections must always stay fp32:
    #   token_embed, pos_embed, *_ln*_w/b, *_bias, lnf_w/b, head_weight
    always_fp32_prefixes = [
        "token_embed", "pos_embed",
        "_ln1_w", "_ln1_b", "_ln2_w", "_ln2_b",
        "_q_bias", "_k_bias", "_v_bias", "_o_bias",
        "_ff1_bias", "_ff2_bias",
        "lnf_w", "lnf_b", "head_weight",
    ]

    for bits in [8, 4]:
        manifest, _ = serialize_tiny(bits)
        for name, sec in manifest["sections"].items():
            is_fp32_section = any(
                name == p or name.endswith(p) or name.startswith(p)
                for p in always_fp32_prefixes
            )
            if is_fp32_section:
                check(sec["dtype"] == "float32",
                      f"bits={bits}: '{name}' must be fp32 (got {sec['dtype']})")


def test_4bit_grouped_output_size():
    """Per-group 4-bit: packed bytes same size as per-tensor; scales array has n_groups entries."""
    torch.manual_seed(7)
    t = torch.randn(256)  # 256 values, group_size=32 → 8 groups
    packed, scales = quantize_4bit_grouped(t, group_size=32)
    check(len(packed) == 128, f"packed bytes: expected 128, got {len(packed)}")
    check(len(scales) == 8, f"n_groups: expected 8, got {len(scales)}")


def test_4bit_grouped_better_accuracy_than_per_tensor():
    """Per-group quantization has lower max reconstruction error than per-tensor."""
    torch.manual_seed(13)
    # Weight with very different magnitudes in two halves — worst case for per-tensor
    t = torch.cat([torch.randn(128) * 10.0, torch.randn(128) * 0.01])

    # Per-tensor
    packed_pt, scale_pt = quantize_4bit(t)
    raw_pt = np.frombuffer(packed_pt, dtype=np.uint8)
    recon_pt = np.array([(((b >> 4) & 0xF) - 8) * scale_pt for b in raw_pt] +
                        [(( b       & 0xF) - 8) * scale_pt for b in raw_pt]).ravel()[:len(t)]

    # Per-group (group_size=32)
    packed_pg, scales_pg = quantize_4bit_grouped(t, group_size=32)
    # Reconstruct per-group
    t_np = t.numpy()
    recon_pg = np.zeros(len(t_np))
    for g in range(len(scales_pg)):
        start, end = g * 32, min((g + 1) * 32, len(t_np))
        n = end - start
        byte_start = start // 2
        byte_end = (end + 1) // 2
        raw = np.frombuffer(packed_pg[byte_start:byte_end], dtype=np.uint8)
        vals = []
        for b in raw:
            vals.append((b >> 4) & 0xF)
            vals.append(b & 0xF)
        for i in range(n):
            recon_pg[start + i] = (vals[i] - 8) * scales_pg[g]

    err_pt = float(np.max(np.abs(t_np - recon_pt[:len(t_np)])))
    err_pg = float(np.max(np.abs(t_np - recon_pg)))
    check(err_pg < err_pt,
          f"grouped error ({err_pg:.4f}) should be < per-tensor error ({err_pt:.4f})")


def test_4bit_grouped_zeros():
    """Zero tensor → reconstructed values are all zero (nibbles encode 0 as 8, not 0)."""
    t = torch.zeros(64)
    packed, scales = quantize_4bit_grouped(t, group_size=32)
    # Reconstruct and verify all zeros
    vals = []
    for b in packed:
        vals.append(((b >> 4) & 0xF) - 8)
        vals.append((b & 0xF) - 8)
    recon = [v * scales[i // 32] for i, v in enumerate(vals[:64])]
    check(all(abs(r) < 1e-6 for r in recon), "zero tensor: reconstructed values should all be ~0")


def test_4bit_grouped_group_size_divides_evenly():
    """Works when tensor size is not a multiple of group_size."""
    t = torch.randn(100)  # 100 values, group_size=32 → 4 groups (last group has 4 values)
    packed, scales = quantize_4bit_grouped(t, group_size=32)
    check(len(scales) == 4, f"expected 4 groups, got {len(scales)}")
    check(len(packed) == 50, f"expected 50 bytes, got {len(packed)}")


# ─── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    test_manifest_structure()
    test_compression_ratios()
    test_fp32_roundtrip()
    test_8bit_quantization_accuracy()
    test_4bit_quantization_accuracy()
    test_zeros_stay_zero()
    test_mixed_precision_layout()
    test_4bit_grouped_output_size()
    test_4bit_grouped_better_accuracy_than_per_tensor()
    test_4bit_grouped_zeros()
    test_4bit_grouped_group_size_divides_evenly()

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


def test_modern_arch_serializes_without_error():
    """TinyTransformerModern checkpoint should serialize to valid bin/json."""
    import tempfile, os
    sys.path.insert(0, str(ROOT / "py"))
    from train_transformer import TinyTransformerModern
    import torch

    model = TinyTransformerModern(vocab_size=16, d_model=8, n_heads=2,
                                   n_layers=1, d_ff=32, max_len=8,
                                   use_rope=False)
    with tempfile.TemporaryDirectory() as td:
        # Save a minimal checkpoint
        ckpt = {
            'model_state': model.state_dict(),
            'architecture': {
                'arch': 'modern', 'vocab_size': 16, 'd_model': 8,
                'n_heads': 2, 'n_layers': 1, 'd_ff': 32, 'max_len': 8,
            },
            'best_val_loss': 0.5,
            'epoch': 10,
        }
        ckpt_path = os.path.join(td, 'test_modern.pt')
        torch.save(ckpt, ckpt_path)

        from serialize import serialize
        prefix = os.path.join(td, 'model_modern')
        serialize(ckpt_path, prefix, weight_bits=8)
        check(os.path.exists(f"{prefix}.bin"), "modern .bin missing")
        check(os.path.exists(f"{prefix}.json"), "modern .json missing")


def test_modern_arch_swiglu_sections_present():
    """Modern arch should have ff_gate weight sections (SwiGLU)."""
    import tempfile, os, json
    sys.path.insert(0, str(ROOT / "py"))
    from train_transformer import TinyTransformerModern
    import torch

    model = TinyTransformerModern(vocab_size=16, d_model=8, n_heads=2,
                                   n_layers=1, d_ff=32, max_len=8,
                                   use_rope=False, use_swiglu=True)
    with tempfile.TemporaryDirectory() as td:
        ckpt = {
            'model_state': model.state_dict(),
            'architecture': {
                'arch': 'modern', 'use_swiglu': True,
                'vocab_size': 16, 'd_model': 8, 'n_heads': 2,
                'n_layers': 1, 'd_ff': 32, 'max_len': 8,
            },
            'best_val_loss': 0.5, 'epoch': 10,
        }
        ckpt_path = os.path.join(td, 'test_swiglu.pt')
        torch.save(ckpt, ckpt_path)
        from serialize import serialize
        prefix = os.path.join(td, 'model_swiglu')
        serialize(ckpt_path, prefix, weight_bits=32)
        manifest = json.loads(open(f"{prefix}.json").read())
        secs = manifest['sections']
        # SwiGLU should produce enc0_ff_gate and enc0_ff_val sections
        check('enc0_ff_gate_weight' in secs, "SwiGLU gate section missing")
        check('enc0_ff_val_weight'  in secs, "SwiGLU val section missing")
        check('enc0_ff2_weight'     in secs, "SwiGLU out section missing")


if __name__ == "__main__":
    test_manifest_structure()
    test_compression_ratios()
    test_fp32_roundtrip()
    test_8bit_quantization_accuracy()
    test_4bit_quantization_accuracy()
    test_zeros_stay_zero()
    test_mixed_precision_layout()
    test_4bit_grouped_output_size()
    test_4bit_grouped_better_accuracy_than_per_tensor()
    test_4bit_grouped_zeros()
    test_4bit_grouped_group_size_divides_evenly()
    test_modern_arch_serializes_without_error()
    test_modern_arch_swiglu_sections_present()

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
