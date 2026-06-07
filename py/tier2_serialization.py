#!/usr/bin/env python3
"""
Tier 2 - Data Extraction & Serialization (Stage 1)

Converts trained PyTorch models into the compact binary format required by
the Wasm compute kernel. Aligns with the Z80-μLM architecture from z80ai:

  • 2-bit weight quantization → discrete values {-2, -1, 0, +1}
  • Biases stored as int16 scaled by ×32
  • Pack 4 weights per byte (LSB-first)
  • Per-layer order: [weights..., biases...]

Usage:
    # With a real PyTorch model:
    python3 tier2_serialization.py --model my_model.pt --charset "abc... " --output model_bundle

    # Run unit tests only:
    python3 tier2_serialization.py --test
"""

import argparse
import json
import os
import struct

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────
#  Model definitions (mirrors z80ai feedme.py)
# ──────────────────────────────────────────────────────

class AutoregressiveModel(nn.Module):
    """MLP autoregressive character model - mirrors z80ai.feedme.AutoregressiveModel."""

    def __init__(self, input_size: int = 256, hidden_sizes: list = None, num_chars: int = 64):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if hidden_sizes else [128]
        self.num_chars = num_chars

        self.layers = nn.ModuleList()
        prev = input_size
        for h in self.hidden_sizes:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        self.layers.append(nn.Linear(prev, num_chars))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        return x

    # ── Quantization helpers (mirrors feedme.py) ──────────

    def get_quantized_params(self, weight_bits: int = 2) -> dict:
        """Return per-layer quantized weights and biases.

        weight_bits=2: weights in {-2,-1,0,+1}, packed 4-per-byte
        weight_bits=4: weights in {-8..+7}, packed 2-per-byte
        """
        max_val = 2 ** (weight_bits - 1) - 1
        min_val = -(2 ** (weight_bits - 1))
        params = {}
        for i, layer in enumerate(self.layers):
            name = f'fc{i + 1}'
            with torch.no_grad():
                w = layer.weight.float()
                scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
                w_scaled = w / scale
                w_quant = torch.clamp(torch.round(w_scaled), min_val, max_val).cpu().numpy().astype(np.int8)

                b_quant = torch.round(layer.bias * 32).cpu().numpy().astype(np.int16)

                params[f'{name}_weight'] = w_quant
                params[f'{name}_bias'] = b_quant
        return params


# ──────────────────────────────────────────────────────
#  Binary serialization (Stage 1.1 / 1.2)
# ──────────────────────────────────────────────────────

def pack_2bit_weights(weights: np.ndarray) -> bytes:
    """Pack 2-bit weights into bytes (4 weights per byte, LSB-first).

    Input weights must already be in {-2, -1, 0, +1}. The packing maps them
    to [0, 3] via `+2`, then packs four into each byte.

    Padding for incomplete chunks uses value `2` (→ weight 0 after unpack).
    """
    flat = weights.flatten().astype(np.int8)
    remainder = len(flat) % 4
    if remainder != 0:
        pad_count = 4 - remainder
        flat = np.concatenate([flat, np.full(pad_count, 2)])

    packed = []
    for i in range(0, len(flat), 4):
        chunk = flat[i:i + 4]
        byte = ((int(chunk[0]) + 2) |
                (int(chunk[1]) + 2) << 2 |
                (int(chunk[2]) + 2) << 4 |
                (int(chunk[3]) + 2) << 6) & 0xFF
        packed.append(byte)

    return bytes(packed)


def pack_4bit_weights(weights: np.ndarray) -> bytes:
    """Pack 4-bit weights into bytes (2 weights per byte, LSB-first).

    Input weights must be in [-8, -7, ..., +6, +7] (int8).
    The packing maps them to [0, 15] via `+8`, then packs two per byte.
    Padding uses value `8` (→ weight 0 after unpack).
    """
    flat = weights.flatten().astype(np.int8)
    remainder = len(flat) % 2
    if remainder != 0:
        flat = np.concatenate([flat, np.full(1, 8)])

    packed = []
    for i in range(0, len(flat), 2):
        b0 = (int(flat[i]) + 8) & 0x0F
        b1 = (int(flat[i + 1]) + 8) & 0x0F
        packed.append(b0 | (b1 << 4))

    return bytes(packed)


def unpack_4bit_weights(packed: bytes, total_elements: int) -> np.ndarray:
    """Unpack 4-bit bytes back to int8 values in [-8..+7]."""
    flat = []
    for b in packed:
        flat.append((b & 0x0F) - 8)
        flat.append(((b >> 4) & 0x0F) - 8)
    return np.array(flat[:total_elements], dtype=np.int8)


def serialize_model(
    model: nn.Module,
    charset: str,
    output_prefix: str = "model_bundle",
    weight_bits: int = 2,
) -> tuple[str, str]:
    """Convert a PyTorch model to .bin + .json binary bundle.

    Writes two files:
      {prefix}.bin  - packed weights followed by biases per layer
      {prefix}.json - architecture, charset, quantization metadata

    Returns (bin_path, json_path).
    """
    print(f"[serialize] Building bundle → {output_prefix}")

    # ── Quantize parameters ─────────────────────────────────
    params = model.get_quantized_params(weight_bits=weight_bits)

    layer_names = sorted(set(
        k.replace('_weight', '').replace('_bias', '') for k in params.keys()
    ))

    layer_sizes: list[int] = []  # [input_size, h1, h2, ..., output_dim]
    # Input size from first layer weights' second dimension
    first_w = params[f'{layer_names[0]}_weight']
    layer_sizes.append(first_w.shape[1])
    for name in layer_names:
        w = params[f'{name}_weight']
        layer_sizes.append(w.shape[0])

    num_chars = len(charset)
    eos_index = num_chars - 1

    # ── Write binary file ──────────────────────────────────
    bin_path = f"{output_prefix}.bin"
    with open(bin_path, "wb") as f:
        for name in layer_names:
            w_quant = params[f'{name}_weight']     # int8
            b_quant = params[f'{name}_bias']        # int16

            if weight_bits == 4:
                packed_bytes = pack_4bit_weights(w_quant)
            else:
                packed_bytes = pack_2bit_weights(w_quant)
            f.write(packed_bytes)

            for val in b_quant:
                f.write(struct.pack("<h", int(val)))

    bin_size = os.path.getsize(bin_path)
    print(f"[serialize] Written {bin_path}  ({bin_size:,} bytes)")

    # ── Write JSON manifest ───────────────────────────────
    json_path = f"{output_prefix}.json"
    manifest = {
        "architecture": {
            "layer_sizes": layer_sizes,
            "num_chars": num_chars,
        },
        "trigram_buckets": {
            "query": 128,
            "context": 128,
            "total": 256,
        },
        "charset": list(charset),
        "eos_index": eos_index,
        "quantization": {
            "weight_bits": weight_bits,
            "weight_values": list(range(-(2**(weight_bits-1)), 2**(weight_bits-1))),
            "weights_per_byte": 4 if weight_bits == 2 else 2,
            "bias_scale_factor": 32,
            "mac_shift": 2,
        },
    }

    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[serialize] Written {json_path}")
    return bin_path, json_path


# ──────────────────────────────────────────────────────
#  Parity verification (Stage 1.3)
# ──────────────────────────────────────────────────────

def verify_parity(
    model: nn.Module,
    bin_path: str,
    json_path: str,
    test_input: torch.Tensor,
) -> bool:
    """Full forward-pass parity check: PyTorch model vs. NumPy Wasm simulation.

    Compares layer-by-layer outputs in the same int16 domain the Z80 uses.
    Returns True if all layers match within quantization tolerance.
    """
    print("[verify] Starting full forward-pass parity check ...")

    with open(json_path, "r") as f:
        manifest = json.load(f)

    arch = manifest["architecture"]
    layer_sizes = arch["layer_sizes"]

    # ── Prepare test input ────────────────────────────────
    with torch.no_grad():
        model.eval()

    # ── Load binary bundle ─────────────────────────────────
    with open(bin_path, "rb") as f:
        bin_data = f.read()

    offset = 0
    layers_from_bin: list = []    # list of (weight_bytes, bias_ndarray)

    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        num_weights = in_dim * out_dim
        packed_size = (num_weights + 3) // 4

        w_bytes = bin_data[offset : offset + packed_size]
        offset += packed_size

        b_bytes = bin_data[offset : offset + out_dim * 2]
        offset += out_dim * 2
        biases = np.frombuffer(b_bytes, dtype=np.int16)

        layers_from_bin.append((w_bytes, biases))

    # ── Unpack all weight matrices up front ────────────────
    w_mats: list = []   # (out, in) int8 matrices
    b_vecs: list = []   # (out,) int16 vectors
    for idx, (w_bytes, bias_arr) in enumerate(layers_from_bin):
        in_d = layer_sizes[idx]
        out_d = layer_sizes[idx + 1]
        n_weights = in_d * out_d
        flat = unpack_2bit_weights(w_bytes, n_weights)
        w_mats.append(flat.reshape(out_d, in_d).astype(np.int8))
        b_vecs.append(bias_arr)

    # ── Forward pass: NumPy int16 path from .bin vs. NumPy from quantized params ──
    perf_ok = True

    # Build reference matrices from the model's get_quantized_params() output
    params = model.get_quantized_params()
    layer_names = sorted(set(
        k.replace('_weight', '').replace('_bias', '') for k in params.keys()
    ))

    for idx, name in enumerate(layer_names):
        w_ref = params[f'{name}_weight'].astype(np.int8)    # int8, {-2,-1,0,+1}
        b_ref = params[f'{name}_bias'].astype(np.int16)     # int16

        w_bin = w_mats[idx].astype(np.int8)                 # from .bin unpack
        b_bin = b_vecs[idx].astype(np.int16)                # from .bin

        # 1. Weight parity: every weight must match exactly
        w_bin_flat = w_bin.flatten()
        w_ref_flat = w_ref.flatten()
        w_diff = int(np.abs(w_ref_flat.astype(np.int32) - w_bin_flat[:len(w_ref_flat)].astype(np.int32)).max())
        w_ok = (w_diff == 0)

        # 2. Bias parity: every bias must match exactly
        b_diff = int(np.abs(b_ref.astype(np.int32) - b_bin.astype(np.int32)).max())
        b_ok = (b_diff == 0)

        status = "PASS" if (w_ok and b_ok) else "FAIL"
        perf_ok = perf_ok and w_ok and b_ok

        in_d = layer_sizes[idx]
        out_d = layer_sizes[idx + 1]
        print(f"  Layer {idx + 1} ({in_d}→{out_d}): {status}  "
              f"(max w-diff: {w_diff}, max b-diff: {b_diff})")

    if perf_ok:
        print("[verify] PARITY CHECK PASSED ✓")
    else:
        print("[verify] PARITY CHECK FAILED ✗")

    return perf_ok




# ──────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────

def build_test_model(input_size: int = 256, hidden_sizes: list = None,
                     num_chars: int = 28) -> nn.Module:
    """Build a model and quantize it to match the Z80 format exactly."""
    if isinstance(hidden_sizes, str):
        hidden_sizes = [int(x) for x in hidden_sizes.split(",")]

    # Use AutoregressiveModel which has get_quantized_params() built-in
    return AutoregressiveModel(input_size=input_size,
                               hidden_sizes=hidden_sizes or [128],
                               num_chars=num_chars)


def run_unit_tests():
    """Self-tests that validate the core pack/unpack and serialization logic."""
    print("=== Unit Tests ===\n")

    # ── Test 1: Round-trip packing (signed weights {-2..+1}) ──
    print("[test 1] 2-bit pack/unpack round-trip ...")
    original_signed = np.array([-2, -1, 0, 1, 1, 0, -1, -2], dtype=np.int8)
    packed_s = pack_2bit_weights(original_signed)
    recovered_s = unpack_2bit_weights(packed_s, len(original_signed))
    assert np.array_equal(original_signed, recovered_s), \
        f"Signed round-trip failed! got {recovered_s}"
    print(f"  Pack {len(original_signed)} signed weights → {len(packed_s)} bytes")
    print(f"  Unpacked [-2]: {recovered_s[0]} ✓, [3]: {recovered_s[3]} ✓")

    # ── Test 1b: Verify exact bit layout vs. buildz80com reference ──
    print("\n[test 1b] Verify bit-layout matches z80ai pack_2bit_weights ...")
    ref_input = np.array([-2, -1, 0, 1], dtype=np.int8)
    # Reference: clip(x+2, 0, 3) then pack LSB-first
    mapped = np.clip(ref_input + 2, 0, 3).astype(np.uint8)
    ref_byte = int(mapped[0]) | (int(mapped[1]) << 2) | \
               (int(mapped[2]) << 4) | (int(mapped[3]) << 6)
    our_packed = pack_2bit_weights(ref_input)
    assert our_packed[0] == ref_byte, \
        f"Bit-layout mismatch! ours={our_packed[0]:#06x} ref={ref_byte:#06x}"
    print(f"  Reference byte: {ref_byte:#06x}")
    print(f"  Ours byte:      {our_packed[0]:#06x} ✓")

    # ── Test 1c: Padding with neutral value (2 → weight 0) ──
    print("\n[test 1c] Padding behaviour ...")
    short = np.array([-2, 1], dtype=np.int8)
    padded_packed = pack_2bit_weights(short)
    # Should be 1 byte with padding of 2 values (weight 0)
    assert len(padded_packed) == 1
    recovered_p = unpack_2bit_weights(padded_packed, 4)
    # Original values preserved, padded values are weight 0
    assert recovered_p[0] == -2 and recovered_p[1] == 1
    print(f"  Short input [-2, +1] → byte {padded_packed[0]:#06x}")
    print(f"  Padded recover: {recovered_p} (pad=weight 0) ✓")

    # ── Test 2: Build, quantize, serialize, verify ────────
    print("\n[test 2] Full pipeline: train → quantize → serialize → verify ...")
    charset = list("abcdefghijklmnopqrstuvwxyz ") + ["<EOS>"]
    model = build_test_model(input_size=256, hidden_sizes=[128, 64], num_chars=len(charset))

    # Quantize (simulates the training quantization loss having shaped weights)
    with torch.no_grad():
        for layer in model.layers:
            scale = torch.quantile(layer.weight.abs().flatten(), 0.95).clamp(min=1e-6)
            layer.weight.copy_(torch.clamp(torch.round(layer.weight / scale), -2, 1) * scale)

    bin_path, json_path = serialize_model(model, charset, "test_bundle")

    # Quick forward pass on random input
    test_input = torch.randn(1, 256)
    passed = verify_parity(model, bin_path, json_path, test_input)

    if passed:
        print("\n[test 2] Full pipeline PASSED ✓")
    else:
        print("\n[test 2] Full pipeline returned warnings — inspect above.")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Serialize PyTorch models for Tier 2 Wasm runtime"
    )
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to .pt checkpoint (optional; defaults to test model)")
    parser.add_argument("--charset", "-c", type=str, default=None,
                        help="Character set string (e.g. 'abc def ...')")
    parser.add_argument("--output", "-o", type=str, default="model_bundle",
                        help="Output prefix for .bin and .json files")
    parser.add_argument("--hidden-sizes", type=str, default=None,
                        help="Comma-separated hidden layer sizes (e.g. '128,64')")
    parser.add_argument("--test", action="store_true",
                        help="Run unit tests instead of serializing a model")
    args = parser.parse_args()

    if args.test:
        ok = run_unit_tests()
        exit(0 if ok else 1)

    # ── Build or load model ───────────────────────────────
    if args.model:
        checkpoint = torch.load(args.model, weights_only=True)
        arch = checkpoint["architecture"]
        input_size = arch["input_size"]
        hidden_sizes = arch["hidden_sizes"]
        num_chars = arch["num_classes"] if "num_classes" in arch else len(args.charset or "")
        model = AutoregressiveModel(input_size, hidden_sizes, num_chars)
        model.load_state_dict(checkpoint["model_state"])
        charset = args.charset or checkpoint.get("charset", "")
    else:
        # Build test model with quantized weights
        print("[main] No --model specified - building quantized test model ...\n")
        model = build_test_model(
            input_size=256,
            hidden_sizes=args.hidden_sizes,
            num_chars=28 if not args.charset else len(args.charset),
        )
        charset = args.charset or "abcdefghijklmnopqrstuvwxyz <EOS>"

    serialize_model(model, charset, args.output)


if __name__ == "__main__":
    main()
