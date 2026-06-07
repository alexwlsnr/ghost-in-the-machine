#!/usr/bin/env python3
"""
Canonical serializer for Ghost in the Machine transformer models.

Reads a PyTorch checkpoint and produces a float32 .bin + .json manifest
suitable for the TS/Wasm inference pipeline. Supports fp32, 8-bit, and
4-bit quantization with per-tensor scales (mixed precision).

The ONLY source-of-truth serializer. Replaces five divergent scripts.

Usage:
  python3 py/serialize.py transformer_model_eos.pt --out dist/model_wisp
  python3 py/serialize.py specter.pt --out dist/model_specter --weight-bits 8
  python3 py/serialize.py specter.pt --out dist/model_specter --weight-bits 4

Mixed-precision layout:
  Quality-sensitive (always fp32): embeddings, head, layer norms, biases.
  Bulk parameters (quantized):     attention Q/K/V/O weights, FFN weights.
"""

import argparse, json, math, struct, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# --- Weight extraction helpers ---

def get_arch(state_dict: dict, checkpoint: dict) -> dict:
    """Return the architecture dict, preferring checkpoint metadata."""
    arch = checkpoint.get("architecture", {})
    if "vocab_size" not in arch and "head.weight" in state_dict:
        arch["vocab_size"] = state_dict["head.weight"].shape[0]
    if "d_model" not in arch and "head.weight" in state_dict:
        arch["d_model"] = state_dict["head.weight"].shape[1]
    return arch


# --- Quantization helpers ---

def quantize_4bit(tensor: torch.Tensor) -> tuple[bytes, float]:
    """Pack a float32 tensor into 4-bit signed ints (-8..7) + a per-tensor scale.

    Packing convention (must match wasm/src/lib.rs matmul_4bit):
      Each byte stores 2 weights.
      High nibble (bits 7-4) = first weight (even index).
      Low  nibble (bits 3-0) = second weight (odd index).
      Stored value = (int_weight + 8) & 0x0F.

    Returns (packed_bytes, scale).
    """
    t = tensor.detach().cpu().to(torch.float32).numpy().ravel()
    absmax = float(np.max(np.abs(t)))
    if absmax < 1e-8:
        absmax = 1.0
    scale = absmax / 7.0
    quant = np.clip(np.round(t / scale), -8, 7).astype(np.int8)
    out = bytearray()
    for i in range(0, len(quant), 2):
        hi = (int(quant[i]) + 8) & 0xF
        lo = (int(quant[i + 1]) + 8) & 0xF if i + 1 < len(quant) else 0
        out.append((hi << 4) | lo)
    return bytes(out), scale


def quantize_8bit(tensor: torch.Tensor) -> tuple[bytes, float]:
    """Quantize a float32 tensor to i8 in [-127, 127] + a per-tensor scale.

    scale = max_abs / 127.0.
    Unpack in kernel: weight_i8 as f32 * scale.

    Returns (raw_bytes, scale).
    """
    t = tensor.detach().cpu().to(torch.float32).numpy().ravel()
    absmax = float(np.max(np.abs(t)))
    if absmax < 1e-8:
        absmax = 1.0
    scale = absmax / 127.0
    quant = np.clip(np.round(t / scale), -127, 127).astype(np.int8)
    return quant.tobytes(), scale


# --- Serializer ---

def serialize(
    checkpoint_path: str,
    out_prefix: str,
    weight_bits: int = 32,
) -> None:
    """Produce <out_prefix>.bin and <out_prefix>.json."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state: dict[str, torch.Tensor] = ck["model_state"]
    arch = get_arch(state, ck)

    d       = arch["d_model"]
    nlayers = arch.get("n_layers", 4)
    vocab   = arch["vocab_size"]
    maxlen  = arch.get("max_len", 64)
    sqrt_d  = math.sqrt(d)

    weight_format = {32: "fp32", 8: "int8", 4: "int4"}[weight_bits]

    sections: dict[str, dict[str, Any]] = {}
    buf = bytearray()

    def add(name: str, tensor: torch.Tensor, *, quantize: bool = False):
        t = tensor.detach().cpu().to(torch.float32)

        if quantize and weight_bits == 4:
            raw, scale = quantize_4bit(t)
            dtype = "int4"
        elif quantize and weight_bits == 8:
            raw, scale = quantize_8bit(t)
            dtype = "int8"
        else:
            raw = t.numpy().tobytes()
            scale = None
            dtype = "float32"

        entry: dict[str, Any] = {
            "offset": len(buf),
            "size":   len(raw),
            "shape":  list(t.shape),
            "dtype":  dtype,
        }
        if scale is not None:
            entry["scale"] = scale
        sections[name] = entry
        buf.extend(raw)

    add("token_embed", state["token_embed.weight"] * sqrt_d)
    add("pos_embed",   state["pos_embed.weight"])

    for li in range(nlayers):
        ls  = f"encoder.layers.{li}"
        pfx = f"enc{li}"

        add(f"{pfx}_ln1_w", state[f"{ls}.norm1.weight"])
        add(f"{pfx}_ln1_b", state[f"{ls}.norm1.bias"])
        add(f"{pfx}_ln2_w", state[f"{ls}.norm2.weight"])
        add(f"{pfx}_ln2_b", state[f"{ls}.norm2.bias"])

        iw = state[f"{ls}.self_attn.in_proj_weight"]
        ib = state[f"{ls}.self_attn.in_proj_bias"]
        for sname, start in [("q", 0), ("k", d), ("v", 2 * d)]:
            add(f"{pfx}_{sname}_weight", iw[start:start + d], quantize=True)
            add(f"{pfx}_{sname}_bias",  ib[start:start + d])

        add(f"{pfx}_o_weight", state[f"{ls}.self_attn.out_proj.weight"], quantize=True)
        add(f"{pfx}_o_bias",   state[f"{ls}.self_attn.out_proj.bias"])

        add(f"{pfx}_ff1_weight", state[f"{ls}.linear1.weight"], quantize=True)
        add(f"{pfx}_ff1_bias",   state[f"{ls}.linear1.bias"])
        add(f"{pfx}_ff2_weight", state[f"{ls}.linear2.weight"], quantize=True)
        add(f"{pfx}_ff2_bias",   state[f"{ls}.linear2.bias"])

    add("lnf_w", state["ln_final.weight"])
    add("lnf_b", state["ln_final.bias"])
    add("head_weight", state["head.weight"])

    bin_path = Path(f"{out_prefix}.bin")
    json_path = Path(f"{out_prefix}.json")

    bin_path.write_bytes(buf)

    manifest = {
        "version":       "4.0",
        "weight_bits":   weight_bits,
        "weight_format": weight_format,
        "architecture":  arch,
        "sections":      sections,
    }
    json_path.write_text(json.dumps(manifest, indent=2))

    total = sum(s["size"] for s in sections.values())
    n_quantized = sum(1 for s in sections.values() if s["dtype"] != "float32")
    print(f"Wrote {bin_path} ({total:,} bytes / {total/1024/1024:.1f} MB)")
    print(f"Wrote {json_path}")
    print(f"Arch: vocab={vocab} d={d} L={nlayers} ctx={maxlen}")
    if weight_bits != 32:
        print(f"{weight_bits}-bit weights: {n_quantized} tensors quantized (embeddings/head/norms/biases remain fp32)")


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical Ghost-in-the-Machine serializer")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--out", "-o", required=True, help="Output prefix (e.g. dist/model_wisp)")
    parser.add_argument("--weight-bits", "--bits", type=int, default=32, choices=[4, 8, 32],
                        help="Weight bit-width: 32 (float32), 8 (int8), or 4 (packed int4)")
    args = parser.parse_args()
    serialize(args.checkpoint, args.out, args.weight_bits)
