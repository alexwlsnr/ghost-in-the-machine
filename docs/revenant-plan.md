# Revenant — Ternary Large Model Design

## Name

**Revenant** — a ghost that returns from apparent death, stronger than before.
Fits the Ghost in the Machine aesthetic and captures what ternary quantization
does: a model with 10× the parameters of Spec512, compressed into roughly the
same browser footprint through 1-bit/ternary weights.

## Why ternary?

Standard int8 quantization (our current approach) gives ~4× compression from
fp32. Ternary quantization ({-1, 0, +1}) gives **20-30×** compression because:

- Each weight needs only ~1.58 bits (log₂(3) = 1.585)
- Matrix multiply becomes addition/subtraction — no floating point multiply
- The zero weight means sparse operations are possible

The maths for Revenant:

| | Spec512 | Revenant |
|---|---|---|
| Params | 27.6M | 300M |
| Bits/weight | 8 | 1.58 |
| Weight storage | 27MB | ~60MB |
| Quality ceiling | limited | dramatically higher |

A 300M param ternary model fits in ~60MB — 2× Spec512's deployed size but
with 10× the parameter count. Parameter count matters enormously for learning
complex patterns, especially for conversational coherence across multiple turns.

## Architecture

Based on BitNet b1.58 (Microsoft, 2024). The key insight: train with ternary
weights from scratch using a straight-through estimator, not post-training
quantization.

```
Params:  300M (target)
d_model: 1024
n_heads: 16
n_layers: 16
d_ff:    4096
ctx:     1024
vocab:   258 (same as all our models)
```

### Weight representation

All linear projection weights (Q, K, V, O, FF1, FF2) are ternary: {-1, 0, +1}.
The training process uses an absmean scaling factor per weight matrix, learned
alongside the ternary values via a straight-through estimator:

```python
def ternarize(w, scale=None):
    if scale is None:
        scale = w.abs().mean()  # absmean scale
    threshold = 0.5 * scale
    return torch.where(w.abs() < threshold, 0,
                       torch.where(w > 0, scale, -scale))
```

Embeddings, layer norms, and biases stay in fp32 (same mixed-precision
strategy as our current quantization).

### Why not standard quantization of a pretrained model?

Post-training ternary quantization of a dense model destroys quality at this
scale. BitNet works because the model **trains knowing it will be ternary** —
the optimizer adapts to the discrete weight space. The analogy is QAT
(quantization-aware training) taken to the extreme.

This means Revenant must be trained from scratch, not derived from an existing
checkpoint.

## Wasm kernel changes needed

### New export: `matmul_ternary`

```rust
// weights: packed ternary — 4 weights per byte (2 bits each: 00=-1, 01=0, 10=+1)
// scale: per-tensor absmean scale (single f32)
// biases, input, output: standard f32
pub fn matmul_ternary(
    weights: *const u8,
    scale: f32,
    biases: *const f32,
    input: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
)
```

The inner loop is pure integer addition — no multiplies:
```rust
for o in 0..out_dim {
    let mut sum = 0i32;
    for i in 0..in_dim {
        let w = unpack_ternary(weights, o * in_dim + i); // -1, 0, or +1
        sum += w * input[i] as i32;  // just ±input[i] or 0
    }
    output[o] = sum as f32 * scale + biases[o];
}
```

### SIMD path: `matmul_ternary_simd`

With SIMD128, process 16 ternary weights per cycle using integer vectors.
The ternary multiply is just conditional negation — much faster than fp32
multiply-accumulate.

## Training approach

### Data requirements

Revenant at 300M params needs substantially more data than Spec512.
Chinchilla optimal: 300M × 20 tokens = 6B tokens. We won't hit that, but
targeting 500M-1B tokens is achievable:

| Source | Estimated tokens |
|---|---|
| SODA full 1.9M dialogues × avg 200 tokens | ~380M |
| Generated scenario dialogues (50K × 300 tokens) | ~15M |
| Multi-turn distilled data | ~5M |
| **Total** | **~400M tokens** |

400M tokens is 2-3 orders of magnitude more than Spec512 v1.1 will see.
At this scale, a 300M ternary model should produce dramatically better
quality than anything in our current stack.

### Training time

With the RTX 5080:
- Spec512 (27.6M, ctx=1024): ~96 sec/epoch on 30K items
- Revenant (300M, ctx=1024): ~10-15× slower per batch = ~15 min/epoch
- At 100 epochs: ~25 hours
- In practice with early stopping and fewer items/epoch: ~15-20 hours

This is a committed overnight run, potentially a weekend. Feasible.

### Training stack changes needed

Current `train_transformer.py` uses standard fp32 weights + QAT as a
regularizer. For Revenant we need:

1. **`TernaryTransformer`** class — mirrors `TinyTransformer` but uses
   `TernaryLinear` layers instead of `nn.Linear`
2. **`TernaryLinear`** — forward pass ternarizes weights via absmean threshold,
   backward pass uses straight-through estimator (STE)
3. **`--model-type ternary`** CLI flag
4. **Serializer** — pack 4 ternary weights per byte, store one scale per layer

### Phase approach

Given training cost, train in two phases:

**Phase 1 (validation run)** — 50M param ternary model on 50K items to
validate the training stack before committing to the full run. 2-3 hours.
If quality looks promising, proceed.

**Phase 2 (full Revenant)** — 300M param model on full SODA + generated data.
15-20 hours.

## Serialization

Revenant 300M ternary:
- Weights: 300M × 1.58 bits ≈ 59MB packed (4 weights/byte = 75MB naive,
  or ~59MB with tight 2-bit packing)
- Scales: 300M/256 ≈ 1.2M f32 values ≈ 5MB
- Embeddings/LN/biases (fp32): ~50MB
- **Total: ~115MB deployed** — slightly above current Spec512 fp32 but with
  10× the parameter count

For browser deployment: the 8-bit serialized Spec512 is 27MB. The ternary
Revenant would be ~60-70MB. Larger but still loadable on desktop browsers.
Mobile on slower connections would feel it.

Mitigation: bf16 embeddings (fp32 → bf16 for token/pos embed saves ~50MB),
bringing total to ~65-70MB.

## Sequencing

1. ✅ Spec512 v1.1 training (underway — same architecture, better data)
2. → Implement `TernaryLinear` + training stack support
3. → Phase 1 validation: 50M ternary model, verify quality
4. → Build full training dataset (full SODA + 50K generated)
5. → Train Revenant 300M
6. → Implement `matmul_ternary` in Rust/Wasm
7. → Serialize + deploy

## Open questions

- **Embedding quantization**: embeddings are 258 × 1024 × 4 bytes = 1MB —
  small enough to leave fp32. But at 300M params the embedding table is still
  relatively small. Leave fp32.
- **Attention scores**: remain fp32 during both training and inference —
  ternary weights only, not activations.
- **Context length**: keep 1024 (same as Spec512). Can increase to 2048 if
  training data supports it, but 1024 is the right starting point.
- **Name for the training dataset**: `revenant_train.txt`
