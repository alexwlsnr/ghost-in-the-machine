# Specification: Tier 2 — "Ghost in the Machine" (Ultra-Lightweight)

## Overview
A near-zero-latency, ultra-low-bandwidth "Expert Widget" designed to run entirely on the client side (Browser / WebAssembly). This tier prioritizes extreme speed and minimal bundle size, providing a stylized intelligence that feels instant.

## Architectural Origin
A direct port of the **Z80-μLM** architecture from [`z80ai/`](../z80/z80ai/). It maintains the same mathematical constraints as the original 1976-era hardware, but executes on modern JS / Wasm engines.

---

## Data Model

### Weight Quantization
Weights are **2-bit, asymmetrically quantized** to four discrete values: **{-2, -1, 0, +1}**. The source model (`get_quantized_params()`) produces these via per-layer scaling:

```python
scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
w_quant = torch.clamp(torch.round(w / scale), -2, 1)
```

For serialization, we must replicate this quantization (or accept pre-quantized weights from a trained checkpoint).

### Bias Quantization
Biases are stored as **16-bit signed integers (`int16`)** scaled by ×32:

```python
b_quant = torch.round(layer.bias * 32).astype(np.int16)
```

This places biases in the same arithmetic domain as the quantized weight products (see MAC below).

### Bit-Packing
Four 2-bit weights are packed into a single byte, LSB-first:

| Byte bits | Weight index |
|-----------|-------------|
| [1:0]     | 0           |
| [3:2]     | 1           |
| [5:4]     | 2           |
| [7:6]     | 3           |

Padding for tensors not divisible by 4 uses the neutral value `2` (which maps to weight `0` after unpacking).

### Binary File Format (`.bin`)
Ordered **per layer**, concatenated in forward pass order:

1. **Weights** — packed 2-bit bytes (`num_out × num_in`, LSB-first packing)
2. **Biases** — little-endian int16 (`num_out` values)

No headers, no length fields. Layout is fully determined by the JSON metadata.

### Metadata Manifest (`.json`)

```jsonc
{
  "architecture": {
    "layer_sizes": [input_dim, h1, h2, ..., output_dim],
    "num_chars": <vocab_size>
  },
  "trigram_buckets": {
    "query": 128,
    "context": 128,
    "total": 256
  },
  "charset": [" ", "A", "B", "...", "\x00"],
  "eos_index": <len(charset) - 1>,
  "quantization": {
    "weight_bits": 2,
    "weight_values": [-2, -1, 0, 1],
    "weights_per_byte": 4,
    "bias_scale_factor": 32
  }
}
```

---

## Compute Pipeline (Z80-μLM Arithmetic)

### Trigram Encoding
Text is hashed into **128 buckets** using polynomial rolling hash. Raw counts (no normalization) are used — the Z80-compatible encoder feeds integer values directly into the network.

### Layer Forward Pass
For each layer, per-neuron:

```
acc = 0
for i in range(num_in):
    w_i = unpacked_2bit_weight(i)           // ∈ {-2, -1, 0, +1}
    acc += x_i * w_i                        // 16-bit signed multiply-accumulate
// Add bias (already in int16, same domain)
acc += bias
// Arithmetic right-shift by 2 (÷4) — two shifts:
acc = acc >> 2                              // sra_h; rr_l × 2
// Store result via IY pointer
out[neuron] = acc
```

### Activation
- **Hidden layers:** ReLU (set negative 16-bit values to zero)
- **Output layer:** No activation — raw logits passed to `argmax`

### Autoregressive Loop
1. Encode query → 128 buckets + context → 128 buckets = 256-dim input
2. Run through MLP layers (Linear → ReLU → Linear → … → Linear)
3. `argmax` over output logits → select character index
4. If EOS, stop; else append character to context and repeat

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Bundle size** | < 50 KB | Wasm module + `.bin` weights + `.json` manifest |
| **Init latency** | < 10 ms | From `fetch()` complete → first inference ready |
| **Inference latency** | < 5 ms / token | Including trigram update |
| **Model size** | ~4–8 KB per layer pair | Heavy on small models (e.g., 256→64→chars) |

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| **Compute kernel** | WebAssembly (C) — MAC loop, 2-bit unpacking |
| **Encoding & orchestration** | TypeScript — trigram hashing, autoregressive loop |
| **Asset loading** | `fetch()` → ArrayBuffer → TypedArray → Wasm memory |
| **Rendering** | HTML5 `<canvas>` + CSS (CRT/scanline filters optional) |

---

## Definition of Done

1. A model trained in Python can be loaded via URL into the browser.
2. The browser-based model produces the **exact same character sequences** as the Python model for the same input.
3. Total initial payload is under 50 KB.
