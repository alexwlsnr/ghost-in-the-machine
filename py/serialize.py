#!/usr/bin/env python3
"""
Canonical serializer for Ghost in the Machine transformer models.

Reads a PyTorch checkpoint and produces a float32 .bin + .json manifest
suitable for the TS/Wasm inference pipeline. Also supports 4-bit quantization
(with per-tensor scales) for the compact path.

The ONLY source-of-truth serializer. Replaces five divergent scripts.

Usage:
  python3 py/serialize.py transformer_model_eos.pt --out dist/model_wisp
  python3 py/serialize.py shade.pt --out dist/model_shade --bits 4
"""

import argparse, json, math, struct, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ─── Weight extraction helpers ─────────────────────────────────────

def get_arch(state_dict: dict, checkpoint: dict) -> dict:
    """Return the architecture dict, preferring checkpoint metadata."""
    arch = checkpoint.get("architecture", {})
    # fallback: infer from shape of head.weight
    if "vocab_size" not in arch and "head.weight" in state_dict:
        arch["vocab_size"] = state_dict["head.weight"].shape[0]
    if "d_model" not in arch and "head.weight" in state_dict:
        arch["d_model"] = state_dict["head.weight"].shape[1]
    return arch


# ─── 4-bit quantization (optional) ──────────────────────────────────

def quantize_4bit(tensor: torch.Tensor) -> tuple[bytes, float]:
    """Pack a float32 tensor into 4-bit signed ints (−8..7) + a per-tensor scale.
    
    Returns (packed_bytes, scale).  Two 4-bit values per byte, big-endian nibbles.
    """
    t = tensor.detach().cpu().to(torch.float32).numpy().ravel()
    absmax = float(np.max(np.abs(t)))
    if absmax < 1e-8:
        absmax = 1.0
    scale = absmax / 7.0
    quant = np.clip(np.round(t / scale), -8, 7).astype(np.int8)
    # pack pairs: upper nibble = quant[i]+8, lower nibble = quant[i+1]+8
    # (+8 shifts range from −8..7 to 0..15 so bit ops are safe)
    out = bytearray()
    for i in range(0, len(quant), 2):
        hi = (int(quant[i]) + 8) & 0xF
        lo = (int(quant[i + 1]) + 8) & 0xF if i + 1 < len(quant) else 0
        out.append((hi << 4) | lo)
    return bytes(out), scale


# ─── Serializer ─────────────────────────────────────────────────────

def serialize(
    checkpoint_path: str,
    out_prefix: str,
    weight_bits: int = 32,   # 32 = float32, 4 = 4-bit packed
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

    sections: dict[str, dict[str, Any]] = {}
    scales:  dict[str, float] = {}
    buf = bytearray()

    def add(name: str, tensor: torch.Tensor, *, quantize: bool = False):
        t = tensor.detach().cpu().to(torch.float32)
        if quantize and weight_bits == 4:
            raw, scale = quantize_4bit(t)
            scales[name] = scale
        else:
            raw = t.numpy().tobytes()
            scales[name] = 1.0   # float32, no scaling
        sections[name] = {
            "offset": len(buf),
            "size":   len(raw),
            "shape":  list(t.shape),
            "dtype":  "int4" if (quantize and weight_bits == 4) else "float32",
        }
        buf.extend(raw)

    # ── Embeddings ─────────────────────────────────────────────
    # token_embed gets sqrt(d) scaling baked in (matching model.forward).
    # pos_embed does NOT (model.forward adds it raw).
    add("token_embed", state["token_embed.weight"] * sqrt_d)
    add("pos_embed",   state["pos_embed.weight"])

    # ── Encoder layers ─────────────────────────────────────────
    for li in range(nlayers):
        ls  = f"encoder.layers.{li}"
        pfx = f"enc{li}"

        # Layer norms (always float32)
        add(f"{pfx}_ln1_w", state[f"{ls}.norm1.weight"])
        add(f"{pfx}_ln1_b", state[f"{ls}.norm1.bias"])
        add(f"{pfx}_ln2_w", state[f"{ls}.norm2.weight"])
        add(f"{pfx}_ln2_b", state[f"{ls}.norm2.bias"])

        # QKV (split from combined in_proj_weight / in_proj_bias)
        iw = state[f"{ls}.self_attn.in_proj_weight"]   # (3*d, d)
        ib = state[f"{ls}.self_attn.in_proj_bias"]     # (3*d,)
        for sname, start in [("q", 0), ("k", d), ("v", 2 * d)]:
            add(f"{pfx}_{sname}_weight", iw[start:start + d], quantize=True)
            add(f"{pfx}_{sname}_bias",  ib[start:start + d])

        # Output projection
        add(f"{pfx}_o_weight", state[f"{ls}.self_attn.out_proj.weight"], quantize=True)
        add(f"{pfx}_o_bias",   state[f"{ls}.self_attn.out_proj.bias"])

        # FFN
        add(f"{pfx}_ff1_weight", state[f"{ls}.linear1.weight"], quantize=True)
        add(f"{pfx}_ff1_bias",   state[f"{ls}.linear1.bias"])
        add(f"{pfx}_ff2_weight", state[f"{ls}.linear2.weight"], quantize=True)
        add(f"{pfx}_ff2_bias",   state[f"{ls}.linear2.bias"])

    # ── Final LN + head ────────────────────────────────────────
    add("lnf_w", state["ln_final.weight"])
    add("lnf_b", state["ln_final.bias"])
    add("head_weight", state["head.weight"], quantize=(weight_bits == 4))

    # ── Write files ────────────────────────────────────────────
    bin_path = Path(f"{out_prefix}.bin")
    json_path = Path(f"{out_prefix}.json")

    bin_path.write_bytes(buf)

    manifest = {
        "version":      "4.0",
        "weight_bits":  weight_bits,
        "architecture": arch,
        "sections":     sections,
        "scales":       scales,
    }
    json_path.write_text(json.dumps(manifest, indent=2))

    total = sum(s["size"] for s in sections.values())
    print(f"Wrote {bin_path} ({total:,} bytes / {total/1024/1024:.1f} MB)")
    print(f"Wrote {json_path}")
    print(f"Arch: vocab={vocab} d={d} L={nlayers} ctx={maxlen}")
    if weight_bits == 4:
        print(f"4-bit weights: {len([k for k,v in scales.items() if v != 1.0])} tensors quantized")


# ─── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical Ghost-in-the-Machine serializer")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--out", "-o", required=True, help="Output prefix (e.g. dist/model_wisp)")
    parser.add_argument("--bits", type=int, default=32, choices=[4, 32],
                        help="Weight bit-width: 32 (float) or 4 (packed)")
    args = parser.parse_args()
    serialize(args.checkpoint, args.out, args.bits)
