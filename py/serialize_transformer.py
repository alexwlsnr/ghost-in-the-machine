#!/usr/bin/env python3
"""
Tier 2.5 Serialization — Tiny Transformer model → .bin / .json bundle

Memory layout:
  [0] Token embeddings:      vocab_size × d_model, 4-bit packed
  [1] Position embeddings:   max_len × d_model, 4-bit packed
  [2..] Per-layer params (for each of n_layers):
    [layer] Q weight + bias
    [layer] K weight + bias
    [layer] V weight + bias
    [layer] O weight + bias
    [layer] FF1 weight + bias
    [layer] FF2 weight + bias
    [layer] LN1 gamma + beta (fp16 as u16)
    [layer] LN2 gamma + beta (fp16 as u16)
  [tail] Final LN gamma + beta (fp16)
  [tail] Output head weight (4-bit packed)

JSON manifest stores the byte offsets of each section.
"""

import json
import struct
import zlib
import numpy as np
from typing import Any, Dict, List, Tuple

PAD_TOKEN = 256
VOCAB_SIZE = 257

# ─── 4-bit packing ──────────────────────────────────────────────────

def pack_4bit(arr: np.ndarray) -> bytes:
    """Pack int8 values (-8..+7) into bytes (2 per byte, LSB-first)."""
    flat = arr.flatten().astype(np.int8)
    if len(flat) % 2:
        flat = np.append(flat, 8)  # pad with zero weight
    packed = bytearray()
    for i in range(0, len(flat), 2):
        b0 = (int(flat[i]) + 8) & 0x0F
        b1 = (int(flat[i + 1]) + 8) & 0x0F
        packed.append(b0 | (b1 << 4))
    return bytes(packed)


def pack_biases(arr: np.ndarray) -> bytes:
    """Pack int16 biases as little-endian bytes."""
    return arr.flatten().astype(np.int16).tobytes()


def pack_fp16(arr: np.ndarray) -> bytes:
    """Pack float16 params as little-endian bytes."""
    return arr.flatten().astype(np.float16).tobytes()


# ─── Serialize ──────────────────────────────────────────────────────

def serialize_transformer(
    model_params: Dict[str, np.ndarray],
    architecture: Dict[str, Any],
    output_prefix: str = "transformer_bundle",
) -> Tuple[str, str]:
    """Serialize a TinyTransformer's quantized params to .bin + .json.

    model_params: output of model.get_quantized_params()
    architecture: dict with d_model, n_heads, n_layers, d_ff, max_len, vocab_size
    """
    d_model = architecture['d_model']
    n_layers = architecture['n_layers']
    max_len = architecture['max_len']
    vocab_size = architecture.get('vocab_size', VOCAB_SIZE)

    bin_parts: List[bytes] = []
    manifest: Dict[str, Any] = {
        "model_type": "tiny_transformer",
        "architecture": architecture,
        "quantization": {"weight_bits": 4, "weights_per_byte": 2, "bias_scale": 32},
        "sections": {},
    }

    def add_section(name: str, data: bytes, shape: List[int], dtype: str):
        offset = sum(len(p) for p in bin_parts)
        bin_parts.append(data)
        manifest["sections"][name] = {
            "offset": offset,
            "size": len(data),
            "shape": shape,
            "dtype": dtype,
        }

    # Token embeddings
    te = model_params['token_embed']
    add_section('token_embed', pack_4bit(te), list(te.shape), 'int8')

    # Position embeddings
    pe = model_params['pos_embed']
    add_section('pos_embed', pack_4bit(pe), list(pe.shape), 'int8')

    # Per-layer params
    for li in range(n_layers):
        pfx = f'enc{li}'
        for name in ['q', 'k', 'v', 'o', 'ff1', 'ff2']:
            w = model_params[f'{pfx}_{name}_weight']
            b = model_params[f'{pfx}_{name}_bias']
            add_section(f'{pfx}_{name}_weight', pack_4bit(w), list(w.shape), 'int8')
            add_section(f'{pfx}_{name}_bias', pack_biases(b), list(b.shape), 'int16')

        # Layer norms (i16, quantized at scale 256)
        for ln_name in ['ln1', 'ln2']:
            for param in ['w', 'b']:
                arr = model_params[f'{pfx}_{ln_name}_{param}']
                add_section(f'{pfx}_{ln_name}_{param}', arr.tobytes(), list(arr.shape), 'int16')

    # Final layer norm (i16)
    for param in ['w', 'b']:
        arr = model_params[f'lnf_{param}']
        add_section(f'lnf_{param}', arr.tobytes(), list(arr.shape), 'int16')

    # Output head
    hw = model_params['head_weight']
    add_section('head_weight', pack_4bit(hw), list(hw.shape), 'int8')

    # Write binary
    bin_path = f"{output_prefix}.bin"
    with open(bin_path, 'wb') as f:
        for part in bin_parts:
            f.write(part)

    # Write manifest
    json_path = f"{output_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(manifest, f)

    # Verify
    bin_size = len(b''.join(bin_parts))
    assert bin_size == sum(len(p) for p in bin_parts), "Binary size mismatch"
    assert bin_size == manifest["sections"]["head_weight"]["offset"] + manifest["sections"]["head_weight"]["size"]

    # Print stats
    weight_bytes = sum(
        s['size'] for name, s in manifest['sections'].items()
        if name.endswith('_weight') or name == 'token_embed' or name == 'pos_embed'
    )
    bias_bytes = sum(
        s['size'] for name, s in manifest['sections'].items()
        if name.endswith('_bias')
    )
    ln_bytes = sum(
        s['size'] for name, s in manifest['sections'].items()
        if 'ln' in name
    )
    print(f"[serialize] Binary: {bin_size:,} bytes ({bin_size/1024:.1f} KB)")
    print(f"  Weights (4-bit): {weight_bytes:,} B ({weight_bytes/1024:.1f} KB)")
    print(f"  Biases (i16):    {bias_bytes:,} B ({bias_bytes/1024:.1f} KB)")
    print(f"  Layer norms:     {ln_bytes:,} B ({ln_bytes/1024:.1f} KB)")

    return bin_path, json_path


# ─── Standalone: convert a PyTorch checkpoint ───────────────────────

if __name__ == '__main__':
    import argparse, torch, sys
    sys.path.insert(0, '.')
    from train_transformer import TinyTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True, help='PyTorch checkpoint')
    parser.add_argument('--output', '-o', default='transformer_bundle', help='Output prefix')
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    arch = ckpt['architecture']
    print(f"  Architecture: {arch}")

    model = TinyTransformer(
        vocab_size=arch['vocab_size'],
        d_model=arch['d_model'],
        n_heads=arch['n_heads'],
        n_layers=arch['n_layers'],
        d_ff=arch['d_ff'],
        max_len=arch['max_len'],
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print("  Quantizing...")
    params = model.get_quantized_params(weight_bits=4)
    print(f"  Got {len(params)} parameter tensors")

    serialize_transformer(params, arch, args.output)
    print(f"Done → {args.output}.bin + {args.output}.json")
