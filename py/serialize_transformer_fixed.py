#!/usr/bin/env python3
"""
Simple serialization for transformer — uses fixed global quantization scales
so the integer matmul >>8 and layer_norm with i16 params work directly
without per-matrix scale correction.

Key insight: instead of per-weight-matrix quantile scaling, use a FIXED
global scale. The training already pushed weights toward quantization levels.
"""

import json, struct, numpy as np
from typing import Any, Dict, List, Tuple

GLOBAL_WEIGHT_SCALE = 0.125  # maps [-1.0, +0.875] → [-8, +7]
ACT_SCALE = 256

def pack_4bit(arr: np.ndarray) -> bytes:
    flat = arr.flatten().astype(np.int8)
    if len(flat) % 2:
        flat = np.append(flat, 8)
    packed = bytearray()
    for i in range(0, len(flat), 2):
        b0 = (int(flat[i]) + 8) & 0x0F
        b1 = (int(flat[i + 1]) + 8) & 0x0F
        packed.append(b0 | (b1 << 4))
    return bytes(packed)

def pack_biases(arr: np.ndarray) -> bytes:
    return arr.flatten().astype(np.int16).tobytes()

def serialize_transformer(
    model_params: Dict[str, np.ndarray],
    architecture: Dict[str, Any],
    output_prefix: str = "transformer_bundle",
) -> Tuple[str, str]:
    d_model = architecture['d_model']
    n_layers = architecture['n_layers']
    vocab_size = architecture.get('vocab_size', 257)

    bin_parts: List[bytes] = []
    manifest: Dict[str, Any] = {
        "model_type": "tiny_transformer",
        "architecture": architecture,
        "quantization": {
            "weight_bits": 4,
            "weights_per_byte": 2,
            "weight_scale": GLOBAL_WEIGHT_SCALE,
            "bias_scale": 32,
            "activation_scale": ACT_SCALE,
            "ln_scale": 256,
        },
        "sections": {},
    }

    def add_section(name: str, data: bytes, shape: List[int], dtype: str):
        offset = sum(len(p) for p in bin_parts)
        bin_parts.append(data)
        manifest["sections"][name] = {
            "offset": offset, "size": len(data),
            "shape": shape, "dtype": dtype,
        }

    for name in ['token_embed', 'pos_embed', 'head_weight']:
        arr = model_params[name]
        add_section(name, pack_4bit(arr), list(arr.shape), 'int8')

    for li in range(n_layers):
        pfx = f'enc{li}'
        for name in ['q', 'k', 'v', 'o', 'ff1', 'ff2']:
            for suffix, arr in [('_weight', model_params[f'{pfx}_{name}_weight']),
                                 ('_bias', model_params[f'{pfx}_{name}_bias'])]:
                key = f'{pfx}_{name}{suffix}'
                data = pack_4bit(arr) if 'weight' in suffix else pack_biases(arr)
                add_section(key, data, list(arr.shape), 'int8' if 'weight' in suffix else 'int16')

        for ln_name in ['ln1', 'ln2']:
            for param in ['w', 'b']:
                key = f'{pfx}_{ln_name}_{param}'
                arr = model_params[key]
                add_section(key, arr.tobytes(), list(arr.shape), 'int16')

    for param in ['w', 'b']:
        key = f'lnf_{param}'
        add_section(key, model_params[key].tobytes(), list(model_params[key].shape), 'int16')

    bin_path = f"{output_prefix}.bin"
    with open(bin_path, 'wb') as f:
        for part in bin_parts:
            f.write(part)
    json_path = f"{output_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(manifest, f)

    print(f"[serialize] Binary: {sum(len(p) for p in bin_parts):,} bytes")
    return bin_path, json_path


if __name__ == '__main__':
    import argparse, torch, sys
    sys.path.insert(0, '.')
    from train_transformer import TinyTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--output', '-o', default='transformer_bundle')
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    arch = ckpt['architecture']

    model = TinyTransformer(
        vocab_size=arch['vocab_size'], d_model=arch['d_model'],
        n_heads=arch['n_heads'], n_layers=arch['n_layers'],
        d_ff=arch['d_ff'], max_len=arch['max_len'],
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print("  Extracting params with fixed global scale...")
    params = {}

    def quant_w(w):
        return torch.clamp(torch.round(w / GLOBAL_WEIGHT_SCALE), -8, 7) \
            .detach().cpu().numpy().astype(np.int8)

    def quant_b(b):
        return torch.round(b * 32).detach().cpu().numpy().astype(np.int16)

    def quant_ln(t):
        return torch.round(t * 256).detach().cpu().numpy().astype(np.int16)

    params['token_embed'] = quant_w(model.token_embed.weight)
    params['pos_embed'] = quant_w(model.pos_embed.weight)

    for li, layer in enumerate(model.encoder.layers):
        pfx = f'enc{li}'
        d = arch['d_model']
        in_w = layer.self_attn.in_proj_weight
        in_b = layer.self_attn.in_proj_bias
        for name, start in [('q', 0), ('k', d), ('v', 2*d)]:
            params[f'{pfx}_{name}_weight'] = quant_w(in_w[start:start+d])
            params[f'{pfx}_{name}_bias'] = quant_b(in_b[start:start+d])

        params[f'{pfx}_o_weight'] = quant_w(layer.self_attn.out_proj.weight)
        params[f'{pfx}_o_bias'] = quant_b(layer.self_attn.out_proj.bias)

        params[f'{pfx}_ff1_weight'] = quant_w(layer.linear1.weight)
        params[f'{pfx}_ff1_bias'] = quant_b(layer.linear1.bias)
        params[f'{pfx}_ff2_weight'] = quant_w(layer.linear2.weight)
        params[f'{pfx}_ff2_bias'] = quant_b(layer.linear2.bias)

        for ln_name, ln in [('ln1', layer.norm1), ('ln2', layer.norm2)]:
            params[f'{pfx}_{ln_name}_w'] = quant_ln(ln.weight)
            params[f'{pfx}_{ln_name}_b'] = quant_ln(ln.bias)

    params['lnf_w'] = quant_ln(model.ln_final.weight)
    params['lnf_b'] = quant_ln(model.ln_final.bias)
    params['head_weight'] = quant_w(model.head.weight)

    serialize_transformer(params, arch, args.output)
    print(f"Done → {args.output}.bin + {args.output}.json")
