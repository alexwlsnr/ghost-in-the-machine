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
    # Vocab/d_model fallback for classic checkpoints without full arch tag
    head = state_dict.get("head.weight") if "head.weight" in state_dict else state_dict.get("token_embed.weight")
    if head is not None:
        if "vocab_size" not in arch: arch["vocab_size"] = head.shape[0]
        if "d_model"    not in arch: arch["d_model"]    = head.shape[1]
    # Propagate modern arch flags from checkpoint to manifest
    if arch.get("arch") == "modern":
        for flag in ("use_swiglu", "use_rope", "use_rmsnorm", "tie_weights"):
            if flag not in arch:
                arch[flag] = True
    if arch.get("arch") == "ternary_modern":
        arch.setdefault("use_rope",    True)
        arch.setdefault("use_rmsnorm", True)
        arch.setdefault("use_swiglu",  arch.get("ffn_type", "swiglu") == "swiglu")
        arch.setdefault("tie_weights", True)
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


def quantize_4bit_grouped(
    tensor: torch.Tensor,
    group_size: int = 32,
) -> tuple[bytes, list[float]]:
    """Pack a float32 tensor into 4-bit ints with per-group scales.

    Same nibble packing as quantize_4bit(), but one scale per group_size
    values instead of one per tensor. Dramatically reduces quantization error
    when weights span very different magnitudes (e.g. large FFN layers).

    Returns (packed_bytes, scales_list) where len(scales_list) == ceil(n/group_size).
    The scales list is stored as a separate float32 array in the manifest.
    """
    t = tensor.detach().cpu().to(torch.float32).numpy().ravel()
    n = len(t)
    n_groups = (n + group_size - 1) // group_size
    out = bytearray()
    scales: list[float] = []

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n)
        chunk = t[start:end]

        absmax = float(np.max(np.abs(chunk)))
        if absmax < 1e-8:
            absmax = 1.0
        scale = absmax / 7.0
        scales.append(scale)

        quant = np.clip(np.round(chunk / scale), -8, 7).astype(np.int8)
        for i in range(0, len(quant), 2):
            hi = (int(quant[i]) + 8) & 0xF
            lo = (int(quant[i + 1]) + 8) & 0xF if i + 1 < len(quant) else 0
            out.append((hi << 4) | lo)

    return bytes(out), scales


def quantize_ternary(tensor: torch.Tensor) -> tuple[bytes, float]:
    """Pack a float32 tensor into 2-bit ternary codes with a single absmean scale.

    Encoding (2 bits per weight, 4 weights per byte, high bits first):
      00 = -1 (negative)   stored as (value / scale ≈ -1)
      01 =  0 (zero)       |value| < 0.5 * scale
      10 = +1 (positive)   value / scale ≈ +1
      11 = unused (treated as 0 on decode)

    Returns (packed_bytes, scale) where scale = absmean(tensor).
    """
    t = tensor.detach().cpu().to(torch.float32).numpy().ravel()
    scale = float(np.mean(np.abs(t)))
    if scale < 1e-8:
        scale = 1.0
    threshold = 0.5 * scale

    # Classify: 0=negative, 1=zero, 2=positive
    codes = np.ones(len(t), dtype=np.uint8)          # default: zero
    codes[t >= threshold]  = 2                        # positive
    codes[t <= -threshold] = 0                        # negative

    n_bytes = (len(t) + 3) // 4
    out = bytearray(n_bytes)
    for i in range(n_bytes):
        byte = 0
        for j in range(4):
            idx = i * 4 + j
            code = int(codes[idx]) if idx < len(t) else 1  # pad with zero
            byte |= (code & 0x3) << (6 - j * 2)
        out[i] = byte
    return bytes(out), scale


def quantize_bf16(tensor: torch.Tensor) -> bytes:
    """Store a float32 tensor as bfloat16 (top 16 bits of each f32 bit pattern).

    No scale needed — bf16 preserves the full fp32 exponent range.
    Conversion in kernel: (u16 as u32) << 16 → f32 bit pattern.
    """
    t = tensor.detach().cpu().to(torch.float32)
    # View as uint32, shift right 16 to get the top 16 bits, store as uint16
    u32 = t.numpy().view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


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

    weight_format = {32: "fp32", 16: "bfloat16", 8: "int8", 4: "int4g"}[weight_bits]

    sections: dict[str, dict[str, Any]] = {}
    buf = bytearray()

    GROUP_SIZE = 32  # weights per scale group for int4 grouped quantization

    def add(name: str, tensor: torch.Tensor, *, quantize: bool = False):
        t = tensor.detach().cpu().to(torch.float32)
        is_ternary_arch = arch.get("arch") in ("ternary", "ternary_modern")

        if quantize and is_ternary_arch:
            raw, scale = quantize_ternary(t)
            scales_raw = None
            dtype = "ternary"
        elif quantize and weight_bits == 4:
            raw, scales = quantize_4bit_grouped(t, group_size=GROUP_SIZE)
            scales_raw = np.array(scales, dtype=np.float32).tobytes()
            dtype = "int4g"
        elif quantize and weight_bits == 8:
            raw, scale = quantize_8bit(t)
            scales_raw = None
            dtype = "int8"
        elif quantize and weight_bits == 16:
            raw = quantize_bf16(t)
            scales_raw = None
            dtype = "bfloat16"
        else:
            raw = t.numpy().tobytes()
            scales_raw = None
            dtype = "float32"

        entry: dict[str, Any] = {
            "offset": len(buf),
            "size":   len(raw),
            "shape":  list(t.shape),
            "dtype":  dtype,
        }
        if dtype == "int4g":
            entry["scales_offset"] = len(buf) + len(raw)
            entry["scales_size"]   = len(scales_raw)
            entry["group_size"]    = GROUP_SIZE
        elif dtype in ("int8", "ternary"):
            entry["scale"] = scale
        sections[name] = entry
        buf.extend(raw)
        if scales_raw is not None:
            buf.extend(scales_raw)

    is_ternary        = arch.get("arch") == "ternary"
    is_modern         = arch.get("arch") == "modern"
    is_ternary_modern = arch.get("arch") == "ternary_modern"
    use_swiglu = arch.get("use_swiglu", True) if is_modern else False
    use_rope   = arch.get("use_rope",   True) if is_modern else False
    use_rmsnorm= arch.get("use_rmsnorm",True) if is_modern else False
    tied       = arch.get("tie_weights",True) if is_modern else False
    ffn_type   = arch.get("ffn_type", "swiglu")  # ternary_modern only

    if is_ternary_modern:
        # TinyTransformerTernaryModern: RoPE (no pos_embed), RMSNorm (no bias),
        # ternary Q/K/V/O (no bias), SwiGLU (w1/w2/w3) or ReLU² (w1/w2), weight-tied head.
        # Embeddings NOT pre-scaled by sqrt(d) — ternary forward does its own scaling.
        add("token_embed", state["token_embed.weight"])
        # no pos_embed — RoPE frequencies are computed at runtime
        for li in range(nlayers):
            pfx = f"enc{li}"
            add(f"{pfx}_ln1_w", state[f"blocks.{li}.norm1.weight"])
            add(f"{pfx}_ln1_b", torch.zeros(d))   # RMSNorm has no bias
            add(f"{pfx}_ln2_w", state[f"blocks.{li}.norm2.weight"])
            add(f"{pfx}_ln2_b", torch.zeros(d))
            for sname, key in [("q","q_proj"),("k","k_proj"),("v","v_proj"),("o","o_proj")]:
                add(f"{pfx}_{sname}_weight", state[f"blocks.{li}.attn.{key}.weight"], quantize=True)
                add(f"{pfx}_{sname}_bias",   torch.zeros(d))  # no bias
            if ffn_type == "relu2":
                add(f"{pfx}_ff1_weight", state[f"blocks.{li}.ff.w1.weight"], quantize=True)
                add(f"{pfx}_ff1_bias",   torch.zeros(state[f"blocks.{li}.ff.w1.weight"].shape[0]))
                add(f"{pfx}_ff2_weight", state[f"blocks.{li}.ff.w2.weight"], quantize=True)
                add(f"{pfx}_ff2_bias",   torch.zeros(d))
            else:  # swiglu
                add(f"{pfx}_ff_gate_weight", state[f"blocks.{li}.ff.w1.weight"], quantize=True)
                add(f"{pfx}_ff_val_weight",  state[f"blocks.{li}.ff.w2.weight"], quantize=True)
                add(f"{pfx}_ff2_weight",     state[f"blocks.{li}.ff.w3.weight"], quantize=True)
                add(f"{pfx}_ff1_bias", torch.zeros(state[f"blocks.{li}.ff.w1.weight"].shape[0]))
                add(f"{pfx}_ff2_bias", torch.zeros(d))
        add("lnf_w", state["ln_final.weight"])
        add("lnf_b", torch.zeros(d))   # RMSNorm has no bias
        add("head_weight", state["token_embed.weight"])  # tied, unscaled

    elif is_ternary:
        # TinyTransformerTernary state dict: blocks.{li}.attn.{q/k/v/o}_proj + ff.{w1/w2}
        # NOTE: ternary forward() does NOT multiply embeddings by sqrt(d_model),
        # so store raw weights here (no pre-scaling unlike classic/modern arch).
        add("token_embed", state["token_embed.weight"])
        add("pos_embed",   state["pos_embed.weight"])
        for li in range(nlayers):
            pfx = f"enc{li}"
            add(f"{pfx}_ln1_w", state[f"blocks.{li}.norm1.weight"])
            add(f"{pfx}_ln1_b", state[f"blocks.{li}.norm1.bias"])
            add(f"{pfx}_ln2_w", state[f"blocks.{li}.norm2.weight"])
            add(f"{pfx}_ln2_b", state[f"blocks.{li}.norm2.bias"])
            for sname, key in [("q","q_proj"),("k","k_proj"),("v","v_proj"),("o","o_proj")]:
                add(f"{pfx}_{sname}_weight", state[f"blocks.{li}.attn.{key}.weight"], quantize=True)
                add(f"{pfx}_{sname}_bias",   state[f"blocks.{li}.attn.{key}.bias"])
            add(f"{pfx}_ff1_weight", state[f"blocks.{li}.ff.w1.weight"], quantize=True)
            add(f"{pfx}_ff1_bias",   state[f"blocks.{li}.ff.w1.bias"])
            add(f"{pfx}_ff2_weight", state[f"blocks.{li}.ff.w2.weight"], quantize=True)
            add(f"{pfx}_ff2_bias",   state[f"blocks.{li}.ff.w2.bias"])
        add("lnf_w",       state["ln_final.weight"])
        add("lnf_b",       state["ln_final.bias"])
        add("head_weight", state["token_embed.weight"])  # tied, unscaled for head projection
    else:
        add("token_embed", state["token_embed.weight"] * sqrt_d)
        if not use_rope:
            add("pos_embed", state["pos_embed.weight"])

        for li in range(nlayers):
            pfx = f"enc{li}"
            if is_modern:
                add(f"{pfx}_ln1_w", state[f"blocks.{li}.norm1.weight"])
                add(f"{pfx}_ln1_b", state[f"blocks.{li}.norm1.bias"] if not use_rmsnorm else torch.zeros(d))
                add(f"{pfx}_ln2_w", state[f"blocks.{li}.norm2.weight"])
                add(f"{pfx}_ln2_b", state[f"blocks.{li}.norm2.bias"] if not use_rmsnorm else torch.zeros(d))
                iw = state[f"blocks.{li}.attn.in_proj.weight"]
                ib = state[f"blocks.{li}.attn.in_proj.bias"]
                for sname, start in [("q", 0), ("k", d), ("v", 2 * d)]:
                    add(f"{pfx}_{sname}_weight", iw[start:start + d], quantize=True)
                    add(f"{pfx}_{sname}_bias",   ib[start:start + d])
                add(f"{pfx}_o_weight", state[f"blocks.{li}.attn.out_proj.weight"], quantize=True)
                add(f"{pfx}_o_bias",   state[f"blocks.{li}.attn.out_proj.bias"])
                if use_swiglu:
                    add(f"{pfx}_ff_gate_weight", state[f"blocks.{li}.ff.w1.weight"], quantize=True)
                    add(f"{pfx}_ff_val_weight",  state[f"blocks.{li}.ff.w2.weight"], quantize=True)
                    add(f"{pfx}_ff2_weight",     state[f"blocks.{li}.ff.w3.weight"], quantize=True)
                    add(f"{pfx}_ff1_bias", torch.zeros(state[f"blocks.{li}.ff.w1.weight"].shape[0]))
                    add(f"{pfx}_ff2_bias", torch.zeros(d))
                else:
                    add(f"{pfx}_ff1_weight", state[f"blocks.{li}.ff.0.weight"], quantize=True)
                    add(f"{pfx}_ff1_bias",   state[f"blocks.{li}.ff.0.bias"])
                    add(f"{pfx}_ff2_weight", state[f"blocks.{li}.ff.2.weight"], quantize=True)
                    add(f"{pfx}_ff2_bias",   state[f"blocks.{li}.ff.2.bias"])
            else:
                ls = f"encoder.layers.{li}"
                add(f"{pfx}_ln1_w", state[f"{ls}.norm1.weight"])
                add(f"{pfx}_ln1_b", state[f"{ls}.norm1.bias"])
                add(f"{pfx}_ln2_w", state[f"{ls}.norm2.weight"])
                add(f"{pfx}_ln2_b", state[f"{ls}.norm2.bias"])
                iw = state[f"{ls}.self_attn.in_proj_weight"]
                ib = state[f"{ls}.self_attn.in_proj_bias"]
                for sname, start in [("q", 0), ("k", d), ("v", 2 * d)]:
                    add(f"{pfx}_{sname}_weight", iw[start:start + d], quantize=True)
                    add(f"{pfx}_{sname}_bias",   ib[start:start + d])
                add(f"{pfx}_o_weight", state[f"{ls}.self_attn.out_proj.weight"], quantize=True)
                add(f"{pfx}_o_bias",   state[f"{ls}.self_attn.out_proj.bias"])
                add(f"{pfx}_ff1_weight", state[f"{ls}.linear1.weight"], quantize=True)
                add(f"{pfx}_ff1_bias",   state[f"{ls}.linear1.bias"])
                add(f"{pfx}_ff2_weight", state[f"{ls}.linear2.weight"], quantize=True)
                add(f"{pfx}_ff2_bias",   state[f"{ls}.linear2.bias"])

        add("lnf_w", state["ln_final.weight"])
        add("lnf_b", state["ln_final.bias"] if not (is_modern and use_rmsnorm) else torch.zeros(d))
        add("head_weight", state["token_embed.weight"] if tied else state["head.weight"])

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

    # Embed BPE tokenizer if checkpoint references one
    tok_file = arch.get('tokenizer_file')
    if tok_file:
        tok_path = Path(tok_file)
        if not tok_path.exists():
            # Try path relative to checkpoint
            tok_path = Path(checkpoint_path).parent / tok_path.name
        if tok_path.exists():
            manifest['tokenizer'] = json.loads(tok_path.read_text())
            print(f"Tokenizer embedded from {tok_path} ({tok_path.stat().st_size // 1024} KB)")
        else:
            print(f"Warning: tokenizer_file {tok_file} not found — manifest will lack tokenizer")

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
    parser.add_argument("--weight-bits", "--bits", type=int, default=32, choices=[4, 8, 16, 32],
                        help="Weight bit-width: 32 (float32), 8 (int8), or 4 (packed int4)")
    args = parser.parse_args()
    serialize(args.checkpoint, args.out, args.weight_bits)
