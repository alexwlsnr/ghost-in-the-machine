#!/usr/bin/env python3
"""
Serialize transformer with per-matrix quantization scales stored in JSON.
The TS/Wasm inference applies these scales after each matmul to compensate
for varying weight ranges across layers and matrix types.
"""

import json, numpy as np, math
from typing import Dict, List, Tuple, Any

def pack_4bit(arr: np.ndarray) -> bytes:
    flat = arr.flatten().astype(np.int8)
    if len(flat) % 2: flat = np.append(flat, 8)
    p = bytearray()
    for i in range(0, len(flat), 2):
        b0 = (int(flat[i]) + 8) & 0x0F
        b1 = (int(flat[i + 1]) + 8) & 0x0F
        p.append(b0 | (b1 << 4))
    return bytes(p)

def serialize(model_params: Dict, scales: Dict, arch: Dict, prefix: str):
    bin_parts = []
    manifest = {
        "model_type": "tiny_transformer",
        "architecture": arch,
        "sections": {},
        "scales": {},
    }

    def add(name: str, data: bytes, shape: List[int], dtype: str):
        off = sum(len(p) for p in bin_parts)
        bin_parts.append(data)
        manifest["sections"][name] = {"offset": off, "size": len(data), "shape": shape, "dtype": dtype}
        if name in scales:
            manifest["scales"][name] = scales[name]

    for key in model_params:
        arr = model_params[key]
        is_weight = 'weight' in key or key in ('token_embed', 'pos_embed')
        data = pack_4bit(arr) if is_weight else arr.flatten().astype(np.float32).tobytes()
        dtype = 'int8' if is_weight else 'float32'
        add(key, data, list(arr.shape), dtype)

    with open(f"{prefix}.bin", 'wb') as f:
        for p in bin_parts: f.write(p)
    with open(f"{prefix}.json", 'w') as f:
        json.dump(manifest, f)

    print(f"Binary: {sum(len(p) for p in bin_parts):,} bytes")
    print(f"Scales: {len(scales)} entries")

if __name__ == '__main__':
    import torch, sys
    sys.path.insert(0, '.')
    from train_transformer import TinyTransformer

    ckpt = torch.load('transformer_model.pt', weights_only=True, map_location='cpu')
    arch = ckpt['architecture']
    model = TinyTransformer(vocab_size=arch['vocab_size'], d_model=arch['d_model'],
        n_heads=arch['n_heads'], n_layers=arch['n_layers'], d_ff=arch['d_ff'], max_len=arch['max_len'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    d = arch['d_model']; dff = arch['d_ff']; nl = arch['n_layers']
    SQRT_D = math.sqrt(d)
    params = {}; scales = {}

    def quant_w(w, name):
        qs = torch.quantile(w.detach().abs().flatten(), 0.95).clamp(min=1e-6).item()
        wq = torch.clamp(torch.round(w.detach() / qs), -8, 7).cpu().numpy().astype(np.int8)
        params[name] = wq; scales[name] = qs

    def quant_b(b, name):
        params[name] = b.detach().cpu().numpy().astype(np.float32)

    def quant_ln(t, name):
        params[name] = t.detach().cpu().numpy().astype(np.float32)

    # Embeddings with sqrt(d) scaling
    quant_w(model.token_embed.weight * SQRT_D, 'token_embed')
    quant_w(model.pos_embed.weight * SQRT_D, 'pos_embed')

    for li, layer in enumerate(model.encoder.layers):
        pfx = f'enc{li}'
        in_w = layer.self_attn.in_proj_weight
        in_b = layer.self_attn.in_proj_bias
        for sname, start in [('q', 0), ('k', d), ('v', 2*d)]:
            quant_w(in_w[start:start+d], f'{pfx}_{sname}_weight')
            quant_b(in_b[start:start+d], f'{pfx}_{sname}_bias')

        quant_w(layer.self_attn.out_proj.weight, f'{pfx}_o_weight')
        quant_b(layer.self_attn.out_proj.bias, f'{pfx}_o_bias')

        quant_w(layer.linear1.weight, f'{pfx}_ff1_weight')
        quant_b(layer.linear1.bias, f'{pfx}_ff1_bias')
        quant_w(layer.linear2.weight, f'{pfx}_ff2_weight')
        quant_b(layer.linear2.bias, f'{pfx}_ff2_bias')

        for ln_name, ln in [('ln1', layer.norm1), ('ln2', layer.norm2)]:
            quant_ln(ln.weight, f'{pfx}_{ln_name}_w')
            quant_ln(ln.bias, f'{pfx}_{ln_name}_b')

    quant_ln(model.ln_final.weight, 'lnf_w')
    quant_ln(model.ln_final.bias, 'lnf_b')
    quant_w(model.head.weight, 'head_weight')

    serialize(params, scales, arch, 'stage3/dist/transformer_model')
