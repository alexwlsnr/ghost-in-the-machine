#!/usr/bin/env python3
"""
Tier 2.5 Serialization v2 — scales tracked per matrix

Fixes the integer/float mismatch by storing per-matrix quantization scales
in the JSON manifest. The TypeScript forward pass applies these scales after
each matmul to compensate for the varying quantization ranges.
"""

import json, struct, numpy as np
from typing import Any, Dict, List, Tuple

PAD_TOKEN = 256
VOCAB_SIZE = 257
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
    param_scales: Dict[str, float],
    architecture: Dict[str, Any],
    output_prefix: str = "transformer_bundle",
) -> Tuple[str, str]:
    d_model = architecture['d_model']
    n_layers = architecture['n_layers']
    max_len = architecture['max_len']
    vocab_size = architecture.get('vocab_size', VOCAB_SIZE)

    bin_parts: List[bytes] = []
    manifest: Dict[str, Any] = {
        "model_type": "tiny_transformer",
        "architecture": architecture,
        "quantization": {
            "weight_bits": 4,
            "weights_per_byte": 2,
            "bias_scale": 32,
            "activation_scale": ACT_SCALE,
        },
        "sections": {},
        "scales": {},
    }

    def add_section(name: str, data: bytes, shape: List[int], dtype: str):
        offset = sum(len(p) for p in bin_parts)
        bin_parts.append(data)
        manifest["sections"][name] = {
            "offset": offset, "size": len(data),
            "shape": shape, "dtype": dtype,
        }
        if name in param_scales:
            manifest["scales"][name] = param_scales[name]

    # Token embeddings
    add_section('token_embed', pack_4bit(model_params['token_embed']),
                list(model_params['token_embed'].shape), 'int8')
    add_section('pos_embed', pack_4bit(model_params['pos_embed']),
                list(model_params['pos_embed'].shape), 'int8')

    # Per-layer params
    for li in range(n_layers):
        pfx = f'enc{li}'
        for name in ['q', 'k', 'v', 'o', 'ff1', 'ff2']:
            for suffix, arr, dtype in [
                ('_weight', model_params[f'{pfx}_{name}_weight'], 'int8'),
                ('_bias', model_params[f'{pfx}_{name}_bias'], 'int16'),
            ]:
                key = f'{pfx}_{name}{suffix}'
                data = pack_4bit(arr) if 'weight' in suffix else pack_biases(arr)
                add_section(key, data, list(arr.shape), dtype)

        # Layer norms (i16)
        for ln_name in ['ln1', 'ln2']:
            for param in ['w', 'b']:
                key = f'{pfx}_{ln_name}_{param}'
                arr = model_params[key]
                add_section(key, arr.tobytes(), list(arr.shape), 'int16')

    # Final layer norm
    for param in ['w', 'b']:
        key = f'lnf_{param}'
        arr = model_params[key]
        add_section(key, arr.tobytes(), list(arr.shape), 'int16')

    # Output head
    add_section('head_weight', pack_4bit(model_params['head_weight']),
                list(model_params['head_weight'].shape), 'int8')

    # Write binary
    bin_path = f"{output_prefix}.bin"
    with open(bin_path, 'wb') as f:
        for part in bin_parts:
            f.write(part)

    # Write manifest
    json_path = f"{output_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(manifest, f)

    bin_size = sum(len(p) for p in bin_parts)
    print(f"[serialize] Binary: {bin_size:,} bytes ({bin_size/1024:.1f} KB)")
    print(f"  Scales tracked: {len(manifest['scales'])} matrices")

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

    print("  Computing quantized params with scales...")

    # Compute params AND per-matrix scales
    params = {}
    scales = {}
    max_val = 7; min_val = -8
    d = arch['d_model']
    dff = arch['d_ff']
    n_layers = arch['n_layers']

    def quant_weight_and_scale(w, name):
        qs = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6).item()
        wq = torch.clamp(torch.round(w / qs), min_val, max_val) \
            .detach().cpu().numpy().astype(np.int8)
        return wq, qs

    def quant_ln(t, name):
        # LN params: scale to fill i16 range
        qs = torch.quantile(t.abs().flatten(), 0.95).clamp(min=1e-6).item()
        tq = torch.round(t / qs * 128).clamp(-32768, 32767) \
            .detach().cpu().numpy().astype(np.int16)
        return tq, qs

    # Embeddings
    w, s = quant_weight_and_scale(model.token_embed.weight, 'token_embed')
    params['token_embed'] = w; scales['token_embed'] = s
    w, s = quant_weight_and_scale(model.pos_embed.weight, 'pos_embed')
    params['pos_embed'] = w; scales['pos_embed'] = s

    for li, layer in enumerate(model.encoder.layers):
        pfx = f'enc{li}'
        d = arch['d_model']

        # Q, K, V
        in_w = layer.self_attn.in_proj_weight
        in_b = layer.self_attn.in_proj_bias
        for name, start in [('q', 0), ('k', d), ('v', 2*d)]:
            w, s = quant_weight_and_scale(in_w[start:start+d], f'{pfx}_{name}_weight')
            params[f'{pfx}_{name}_weight'] = w; scales[f'{pfx}_{name}_weight'] = s
            params[f'{pfx}_{name}_bias'] = torch.round(in_b[start:start+d] * 32) \
                .detach().cpu().numpy().astype(np.int16)

        # Output projection
        w, s = quant_weight_and_scale(layer.self_attn.out_proj.weight, f'{pfx}_o_weight')
        params[f'{pfx}_o_weight'] = w; scales[f'{pfx}_o_weight'] = s
        params[f'{pfx}_o_bias'] = torch.round(layer.self_attn.out_proj.bias * 32) \
            .detach().cpu().numpy().astype(np.int16)

        # FFN
        w, s = quant_weight_and_scale(layer.linear1.weight, f'{pfx}_ff1_weight')
        params[f'{pfx}_ff1_weight'] = w; scales[f'{pfx}_ff1_weight'] = s
        params[f'{pfx}_ff1_bias'] = torch.round(layer.linear1.bias * 32) \
            .detach().cpu().numpy().astype(np.int16)

        w, s = quant_weight_and_scale(layer.linear2.weight, f'{pfx}_ff2_weight')
        params[f'{pfx}_ff2_weight'] = w; scales[f'{pfx}_ff2_weight'] = s
        params[f'{pfx}_ff2_bias'] = torch.round(layer.linear2.bias * 32) \
            .detach().cpu().numpy().astype(np.int16)

        # Layer norms
        for ln_name, ln_layer in [('ln1', layer.norm1), ('ln2', layer.norm2)]:
            for param, tensor in [('w', ln_layer.weight), ('b', ln_layer.bias)]:
                arr, s = quant_ln(tensor, f'{pfx}_{ln_name}_{param}')
                params[f'{pfx}_{ln_name}_{param}'] = arr
                scales[f'{pfx}_{ln_name}_{param}'] = s

    # Final LN
    for param, tensor in [('w', model.ln_final.weight), ('b', model.ln_final.bias)]:
        arr, s = quant_ln(tensor, f'lnf_{param}')
        params[f'lnf_{param}'] = arr; scales[f'lnf_{param}'] = s

    # Head
    w, s = quant_weight_and_scale(model.head.weight, 'head_weight')
    params['head_weight'] = w; scales['head_weight'] = s

    print(f"  Params: {len(params)} tensors, {len(scales)} scales")
    serialize_transformer(params, scales, arch, args.output)
    print(f"Done → {args.output}.bin + {args.output}.json")
