# Bonsai / 1-bit Ternary Viability as Model 4 (Whisper Tier)

**Branch:** `analysis/bonsai-model4`
**Date:** 2026-06-07
**Status:** Research analysis — no source files modified

---

## 1. What Bonsai Actually Is

[PrismML](https://prismml.com) is a Caltech-incubated startup that emerged from stealth in
March 2026 with a $16.25M seed round. Their [Bonsai-demo repo](https://github.com/PrismML-Eng/Bonsai-demo)
ships two model families:

| Family | Bit-width | Scheme | Storage |
|--------|-----------|--------|---------|
| **Bonsai** | 1-bit | True binary: weights ∈ {−1, +1} only | Q1_0 (GGUF), MLX 1-bit |
| **Ternary-Bonsai** | 1.58-bit | Ternary: weights ∈ {−1, 0, +1} | Q2_0 (GGUF 2-bit packed), MLX 2-bit |

Both families ship in 8B, 4B, and 1.7B parameter sizes. The 1-bit Bonsai-8B fits in
**1.15 GB** — roughly 14× smaller than its fp16 equivalent. Ternary-Bonsai occupies
approximately 1.85× the memory of the 1-bit variant (2 bits vs 1 bit per weight).

Whitepapers are included in the repo:
- [`1-bit-bonsai-8b-whitepaper.pdf`](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)
- [`ternary-bonsai-8b-whitepaper.pdf`](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)

Bonsai is a **pre-trained, trained-from-scratch native ultra-low-bit model**, not a
quantization of a larger fp16 checkpoint. All layers — embeddings, attention, FFN, and
the LM head — are natively 1-bit or ternary end-to-end.

The demo repo is a **consumer-facing inference demo only** (llama.cpp + MLX shell
scripts). No training code is provided. The inference stack is:
- llama.cpp fork ([PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp))
  with Q1_0/Q2_0 backends merged or in flight for CPU, Metal, CUDA, Vulkan
- MLX fork for Apple Silicon

Benchmarks claimed: 1-bit Bonsai-8B scores **70.5 average** across MMLU Redux, MuSR,
GSM8K, HumanEval+, IFEval, BFCLv3 — competitive with leading 8B-class fp16 models at
8× the inference speed and 5× lower energy on edge hardware.

---

## 2. What "1-bit" and "1.58-bit" Mean Technically

### 1-bit (Bonsai)

True binary quantization: every linear layer weight is constrained to {−1, +1}. No zero.
This differs from BitNet b1.58. A 1-bit weight requires exactly 1 bit of storage;
8 weights pack into 1 byte. GGUF format `Q1_0` encodes this. A per-group scalar
(group size 128 for Bonsai) restores the correct magnitude during inference.

### 1.58-bit (Ternary-Bonsai, BitNet b1.58-style)

Ternary: weights ∈ {−1, 0, +1}. Information content is log₂(3) ≈ 1.58 bits per weight,
hence the name. Ternary-Bonsai stores these in **Q2_0 format** (2 bits per weight,
groups of 128), which is hardware-friendly but wastes ~0.42 bits/weight vs. optimal
ternary packing. A theoretically optimal ternary pack (base-3 encoding) yields
~5.04 weights/byte vs. 4 weights/byte for Q2_0 — a ~20% improvement that is rarely
implemented in practice due to decode complexity.

### Training scheme (BitNet b1.58 reference, Bonsai proprietary)

Microsoft's BitNet b1.58 (the closest published analog) uses **W1.58A8**: during forward
passes, weights are quantized to {−1, 0, +1} via AbsMean quantization (divide by mean
absolute value, round to nearest ternary value). Activations are quantized to INT8
per-token via AbsMax. Master weights remain in BF16 for gradient updates; ternary weights
exist only in the forward pass. This is true QAT using straight-through estimators for
the non-differentiable rounding step.

PrismML's whitepaper describes results but not their full training pipeline.

---

## 3. Connection to the Project's Existing 2-bit Quantization

The original z80 micro-LLM used **2-bit weights {−2, −1, 0, +1} packed 4-per-byte**,
implemented in `py/archive/tier2_serialization.py`. This is the same physical packing
density as Ternary-Bonsai's Q2_0 (4 weights/byte), but with a different value set:
4 discrete levels instead of 3. The 2-bit z80 format maps:

```
raw value  stored  bit pattern
   −2   →    0    00
   −1   →    1    01
    0   →    2    10
   +1   →    3    11
```

Ternary {−1, 0, +1} uses 3 of those 4 slots (omitting −2). The existing
`pack_2bit_weights()` and `unpack_2bit_weights()` functions in `tier2_serialization.py`
are **directly reusable for ternary** with only a minor value-mapping adjustment.

The current transformer trainer's `compute_quantization_loss()` in `train_transformer.py`
nudges weights toward rounded quantization points:

```python
loss = loss + F.mse_loss(p_scaled, torch.round(p_scaled))
```

For a 4-bit grid this encourages weights toward {−8..+7}. Adapting for ternary QAT
requires only changing the target: `clamp(round(p_scaled), -1, 1)`. The QAT penalty
cadence concern flagged in `multi-model-plan.md` (running `torch.quantile` every step)
applies equally to ternary training.

The current Wasm kernel `wasm/src/lib.rs` uses 4-bit `unpack_weight_f32()` and would
need a parallel 2-bit ternary unpack path — a small self-contained addition.

---

## 4. Size Estimates for a Ternary "Whisper" Tier

A "Whisper" ultra-micro tier below Wisp would logically use:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 128 | Half of Wisp's 256; keeps d_head=64 with 2 heads |
| d_ff | 512 | 4× d_model, consistent with all tiers |
| Layers | 2 | Half of Wisp's 4 |
| Heads | 2 | d_head=64 maintained (matches TS divisibility constraint) |
| ctx | 32 | Half of Wisp's 64 |
| Vocab | 258 | Same as all tiers (bytes 0–255 + PAD + EOS) |

**Parameter breakdown:**

| Component | Formula | Count |
|-----------|---------|-------|
| Token embed | 258 × 128 | 33,024 |
| Pos embed | 32 × 128 | 4,096 |
| Per-layer attn (Q+K+V+O) | 4 × 128² | 65,536 |
| Per-layer FFN (ff1+ff2) | 2 × 128 × 512 | 131,072 |
| Per-layer LN params (×2) | 4 × 128 | 512 |
| × 2 layers | — | 394,240 |
| LN final | 2 × 128 | 256 |
| Head | 258 × 128 | 33,024 |
| **Total** | | **~464,000** |

**Storage at different precisions (linear-layer weights only, ~380K params):**

| Format | Bits/weight | Linear weights | +fixed overhead* | Total |
|--------|-------------|----------------|-----------------|-------|
| fp32 | 32 | ~1.52 MB | ~150 KB | ~1.88 MB |
| fp16 | 16 | ~0.76 MB | ~150 KB | ~0.94 MB |
| int8 | 8 | ~380 KB | ~150 KB | ~530 KB |
| 4-bit (current path) | 4 | ~190 KB | ~150 KB | ~340 KB |
| 2-bit / ternary (Q2_0) | 2 | ~95 KB | ~150 KB | ~245 KB |
| 1-bit binary | 1 | ~48 KB | ~150 KB | ~198 KB |

*Fixed overhead: embeddings (33K × 258 = ~130 KB fp32), biases, LN params — not worth
quantizing at this scale.

**For comparison:**
- Wisp fp32: 13 MB → Wisp 4-bit target: ~1.7 MB
- Whisper fp32: ~1.88 MB → Whisper 4-bit: ~340 KB → Whisper ternary: ~245 KB

The ternary vs 4-bit saving at Whisper scale is **~95 KB** — from ~340 KB to ~245 KB.
That is a 28% reduction on a file that already fits in browser memory in ~1ms. It is not
a meaningful user-facing improvement. The compression payoff from ternary only matters at
billion-parameter scale where linear layers dominate the weight budget.

---

## 5. Technical Comparison to the Current Stack

### The current Wasm kernel

`wasm/src/lib.rs` packs 4-bit weights (2 per byte, values −8..+7 after adding 8),
unpacks them to f32 per element multiplied by `GLOBAL_WEIGHT_SCALE = 0.4`, then runs a
scalar f32 multiply-accumulate loop. The inner loop runs over `in_dim × out_dim` elements
with no SIMD. This produces ~36ms/token on Wisp (d=256, 4 layers).

### Ternary matmul in Wasm

A ternary weight ∈ {−1, 0, +1} means the multiply reduces to a conditional add/subtract:

```
result += w == +1 ? input[i] : (w == -1 ? -input[i] : 0.0)
```

The benefit: no float multiply instruction for any weight. The cost: branch overhead,
or a branchless masking implementation using comparisons.

In Wasm **without SIMD128**, the conditional-add path is not reliably faster than 4-bit
scalar multiply — branch prediction in the Wasm JIT is backend-dependent, and the
unpack step still requires byte reads. No significant speedup is expected.

In Wasm **with SIMD128**, there is a realistic acceleration path: unpack ternary nibbles,
use `i8x16.eq` comparisons to generate masks, apply `v128.and` to mask-select activations,
then accumulate with `f32x4.add`. This avoids float multiplies for the entire matmul and
can process 16 weights per iteration vs. 2 for the current scalar 4-bit loop. The
[r3-engine](https://github.com/r3-engine/r3-engine) attempts this (Rust → `wasm32-unknown-unknown`
+ SIMD128), but its WASM path is currently incomplete — the engine outputs `<unk>` for
all generation tokens, meaning ternary Wasm inference is still an unsolved engineering
problem in 2026.

**Realistic speedup estimate:** A working ternary SIMD128 matmul might be 2–3× faster
than the current scalar 4-bit loop for large matrices. At Whisper scale (128×512 matmul),
the bottleneck shifts to the autoregressive serial loop and JS↔Wasm boundary overhead.
Wall-clock speedup per token: 10–20%, not transformative.

### Browser inference path

Bonsai has a [WebGPU demo](https://huggingface.co/spaces/webml-community/bonsai-webgpu)
and a [Ternary-Bonsai WebGPU demo](https://huggingface.co/spaces/webml-community/bonsai-ternary-webgpu),
but these run on GPU via WebGPU, not the CPU-only Wasm path this project uses. The
project's `wasm32-unknown-unknown` (no_std, no allocator) constraint excludes WebGPU.
**There is no production Wasm CPU inference path for Bonsai or any ternary transformer
in browser environments at the time of this analysis.**

---

## 6. Viability Assessment

### What Bonsai offers that applies here

| Finding | Applicability |
|---------|---------------|
| Proof that native 1-bit/ternary training converges at scale | Encouraging at concept level; irrelevant at <1M params |
| No training code provided | Cannot adopt Bonsai's training recipe |
| Q2_0 packing = same density as archive 2-bit packing | `pack_2bit_weights()` is directly reusable with minor value remapping |
| WebGPU demos exist | Wrong inference backend for this project |
| Q1_0 merged into upstream llama.cpp | Irrelevant — project does not use llama.cpp |

### What Bonsai does NOT offer

- No sub-1B models (smallest is 1.7B; already orders of magnitude larger than Whisper)
- No training code
- No browser/Wasm CPU inference path
- No architecture designed for byte-level vocab=258 tiny transformers

Bonsai demonstrates that native 1-bit training is viable at billion-parameter scale with
full NLP vocabulary (~32K tokens). This does not extrapolate to a 470K parameter byte-
level model. Small models are capacity-constrained — reducing weights from 4-bit (16 levels)
to ternary (3 levels) at sub-1M params will degrade quality substantially. Bonsai's own
quality results require billions of parameters and trillions of training tokens to manifest.

### The z80 prior art is more relevant than Bonsai

The original z80 micro-LLM used 2-bit {−2,−1,0,+1} weights in a byte-level MLP
successfully. That is closer to what a Whisper experiment would involve than anything
in the Bonsai repo. The lesson from z80 is that 2-bit packing works for small MLPs.
The transformer at Wisp's 3.3M params already pushes what 4-bit QAT tolerates. Adding
ternary constraints to a 470K-param model that will already be near its capacity limit
is a quality risk with unclear upside.

---

## 7. Recommendation

**Do NOT add a Model 4 "Whisper" ternary tier at this time.**

### Primary reasons

1. **Marginal size savings.** Going from 4-bit (~340 KB) to ternary (~245 KB) at 470K
   params saves ~95 KB. This is not a user-facing improvement for a model that already
   loads in under a second. The compression justification that made Bonsai interesting
   (1.15 GB vs 16 GB) does not apply here.

2. **Quality tradeoff is unfavorable at this scale.** Reducing from 16 quantization
   levels (4-bit) to 3 (ternary) is a far larger relative constraint at 470K params than
   at 8B params. The model is capacity-starved at Whisper size regardless of bit-width;
   ternary makes this worse with no compensating architecture benefit.

3. **No Wasm speed benefit in practice.** Without a working ternary SIMD128 kernel (none
   exists in production as of June 2026), there is no inference speed argument. The KV
   cache (Phase 3 roadmap) will deliver far more speedup than ternary matmul ever would
   at this parameter count.

4. **Bonsai is not applicable to this project.** It is a consumer inference demo for
   1.7B–8B models with GPU/Metal as the target, no training code, no byte-level
   architecture, and no Wasm CPU path. The repo's engineering and research is orthogonal
   to what this project needs.

5. **Higher-priority roadmap work exists.** The existing 8-phase plan covers
   infrastructure, data expansion, KV caching, quantization paths, and the Specter tier.
   A fourth model tier adds training burden, eval complexity, new UI/CI artifacts, and a
   new model file to ship — all before the current three tiers are fully functional.

### When to revisit

- After Specter ships and Wisp 4-bit lands at ~1.7 MB, if there is genuine demand for a
  sub-300 KB bundle, Whisper becomes worth a time-boxed experiment.
- After the Wasm kernel gains SIMD128 (planned Phase 3 follow-on) and ternary matmul
  benchmarks confirm >2× speedup at Wisp-class dimensions.
- The engineering cost at that point is low: ~50 lines of Rust for the 2-bit unpack path,
  a QAT target adjustment in `train_transformer.py`, and reusing `pack_2bit_weights()`
  from the archive with value remapping.

---

## 8. Hypothetical Tier Spec (For Future Reference Only)

If a Whisper tier were added, the proposed spec would be:

| Name | d_model | d_ff | Layers | Heads | ctx | Params | fp32 | 4-bit | 2-bit ternary |
|------|---------|------|--------|-------|-----|--------|------|-------|---------------|
| **Whisper** | 128 | 512 | 2 | 2 | 32 | ~470K | ~1.88 MB | ~340 KB | ~245 KB |

Infrastructure changes required (all small, self-contained):

- `wasm/src/lib.rs`: add `matmul_2bit_ternary_f32()` entry point; unpack maps `{0,1,2}`
  stored → `{-1,0,+1}` × per-tensor scale
- `py/train_transformer.py`: extend `compute_quantization_loss()` to accept `weight_bits=2`
  and clamp QAT target to `clamp(round(p_scaled), -1, 1)`
- Canonical serializer (Phase 1 prerequisite): add 2-bit ternary pack path reusing
  `pack_2bit_weights()` from `py/archive/tier2_serialization.py` with value remapping
  from {−2,−1,0,+1} → {−1,0,+1}
- `ts/src/tier2_transformer.ts`: weight-format dispatch based on manifest `weight_bits`
  field (already parameterized by `weight_bits` in the Python serializer)
- `models.json`: add Whisper entry; UI model switcher shows "Whisper (ultra-micro)"

No Wasm ABI changes affect the other three tiers.

---

## 9. Summary

| Criterion | Assessment |
|-----------|------------|
| Is Bonsai's architecture applicable here? | No — billion-param models, full NLP vocab, no byte-level design |
| Does Bonsai provide training code? | No — inference demo only |
| Is there a Wasm CPU inference path for Bonsai/ternary? | No — WebGPU only for browser; llama.cpp for native CPU |
| Does ternary improve Wasm inference speed meaningfully? | Not without SIMD128 kernel work; marginal even then at Whisper scale |
| Does ternary save meaningful space vs 4-bit at 470K params? | No — ~95 KB saving on a ~340 KB file |
| Can the existing z80 2-bit infrastructure be adapted? | Yes — minimal changes, low engineering cost |
| Is quality tradeoff acceptable for a new tier? | Unlikely at <1M params; no evidence it would be coherent |
| **Verdict** | **Do not add Model 4 now; revisit after Specter ships and SIMD lands** |

---

## References

- [PrismML Bonsai-demo repo](https://github.com/PrismML-Eng/Bonsai-demo)
- [PrismML announcement: 1-bit Bonsai](https://prismml.com/news/bonsai-8b)
- [The Register: PrismML debuts 1-bit LLM](https://www.theregister.com/2026/04/04/prismml_1bit_llm/)
- [Bonsai 1-bit — technical guide](https://getdeploying.com/guides/bonsai-1bit-llm)
- [BitNet b1.58 2B4T Technical Report](https://arxiv.org/html/2504.12285v1) — Microsoft, April 2025
- [Bonsai WebGPU demo](https://huggingface.co/spaces/webml-community/bonsai-webgpu)
- [Ternary-Bonsai WebGPU demo](https://huggingface.co/spaces/webml-community/bonsai-ternary-webgpu)
- [r3-engine: Safe Rust ternary inference → WASM SIMD128](https://github.com/r3-engine/r3-engine) (incomplete)
- [microsoft/BitNet inference framework](https://github.com/microsoft/BitNet)
- Project: `py/archive/tier2_serialization.py` — 2-bit {−2,−1,0,+1} pack/unpack (prior art)
- Project: `py/archive/feedme.py` — original MLP QAT training
- Project: `wasm/src/lib.rs` — current 4-bit Wasm matmul with `GLOBAL_WEIGHT_SCALE`
- Project: `docs/multi-model-plan.md` — 3-tier architecture and roadmap
