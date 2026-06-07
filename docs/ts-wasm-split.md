# TS → Wasm Split: Analysis and Recommendations

**Branch:** `analysis/ts-wasm-split`  
**Date:** 2026-06-07  
**Architecture under analysis:** Specter (d=768, n_heads=12, d_head=64, n_layers=8, d_ff=3072, ctx=256, vocab=258)

---

## 1. What currently lives where

### Rust/Wasm kernel (`wasm/src/lib.rs`, ~228 lines)

| Export | Purpose | Lines |
|---|---|---|
| `matmul_f32w` | Float32 matmul with bias (row-major weights) | 206–227 |
| `softmax_f32` | Row softmax in-place | 82–108 |
| `softmax_causal_f32` | Full [seq×seq] causal mask + per-row softmax | 112–149 |
| `layer_norm_f32` | Pre-norm LN with gamma/beta | 154–178 |
| `add_vec_f32` | Fused elementwise add (residual) | 183–189 |
| `relu_f32` | Elementwise ReLU | 192–197 |
| `scale_f32` | Scalar scale in-place | 200–203 |
| `matmul_f32` | **DEAD** — 4-bit packed matmul with bias | 25–50 |
| `matmul_no_bias_f32` | **DEAD** — 4-bit packed matmul, no bias | 54–77 |

### TypeScript orchestrator (`ts/src/tier2_transformer.ts`)

| Section | Location | What it does |
|---|---|---|
| Embedding lookup | `forward()` lines 110–116 | JS loops: `emb[p*d+j] = teW[tid*d+j] + peW[p*d+j]` |
| LN + QKV per position | Lines 122–130 | **Per-position Wasm calls** — `layer_norm_f32` then `matmul_f32w` ×3 for each of T positions |
| Q·Kᵀ dot products | Lines 139–146 | **Pure JS triple-nested loop** — `dh`-wide dot product for each (qi, kj) pair |
| Causal mask + softmax | Line 148 | Wasm call (`softmax_causal_f32`) on the full [seq×seq] scores buffer |
| Weighted V sum | Lines 149–155 | **Pure JS triple-nested loop** — accumulates attn-weighted V per head |
| Output projection | Lines 158–160 | Per-position Wasm calls — `matmul_f32w` for each of T positions |
| Residual add | Line 161 | Wasm call (`add_vec_f32` on seq×d) |
| FFN (LN + ff1 + ReLU + ff2) | Lines 163–173 | Per-position Wasm calls — 4 calls per position |
| Final LN + head | Lines 177–186 | Wasm calls (last position only) |
| Sampling loop | Lines 200–223 | JS: softmax with temperature, multinomial draw |

---

## 2. FLOP breakdown for Specter at token step T

All FLOP counts are multiply-accumulates (MACs). Specter parameters: d=768, dh=64, nh=12, nl=8, d_ff=3072.

### Per-layer costs at T positions (generating token T)

**QKV projections (Wasm `matmul_f32w`):**  
3 × T × d² = 3 × T × 589,824 MACs

**O projection (Wasm `matmul_f32w`):**  
T × d² = T × 589,824 MACs

**FFN (Wasm `matmul_f32w` ×2 + `relu_f32`):**  
T × 2 × d × d_ff = T × 4,718,592 MACs

**Attention Q·Kᵀ (JS loops):**  
nh × dh × T(T+1)/2 = 768 × T(T+1)/2 MACs  
*(causal: only lower triangle computed)*

**Weighted V sum (JS loops):**  
nh × dh × T × T = 768 × T² MACs  
*(iterates full T×T even though upper triangle is already zeroed)*

### Absolute counts at representative T values

| T | Wasm MACs | JS attn MACs | JS wall-clock fraction* |
|---|---|---|---|
| 10 | 566M | 0.95M | ~0.5% |
| 50 | 2,831M | 23M | ~14% |
| 100 | 5,663M | 92M | ~25% |
| 150 | 8,494M | 208M | ~35% |
| 200 | 11,325M | 369M | ~43% |
| 255 | 14,439M | 600M | ~45% |

*Wall-clock fraction estimated assuming Wasm compiled loops run ~20× faster than JS interpreter loops for tight numeric code — a conservative but realistic ratio for V8. Raw FLOP fraction understates the problem significantly because the JS interpreter cannot auto-vectorize or pipeline the inner loops.*

**Key insight:** By T=255 (max context), the attention JS loops consume roughly 45% of estimated wall-clock time despite being only 4% of raw FLOPs. This gap widens as context grows because attention scales O(T²) while the matmuls scale O(T).

### O(T²) recompute — the dominant cost driver

The current forward pass recomputes **full** Q·K·V attention for all T positions on every generation step. There is no KV cache. The per-step cost grows quadratically:

| T | Full-recompute JS attn MACs | Incremental (KV cache, 1 new query) | Speedup from KV cache |
|---|---|---|---|
| 50 | 23M | 307K | ~75× |
| 100 | 92M | 614K | ~150× |
| 200 | 369M | 1.23M | ~300× |
| 255 | 600M | 1.57M | ~383× |

The KV cache also eliminates O(T²) recompute on the Wasm matmul side: QKV projections drop from T×3d² to 1×3d² per step (100×–255× reduction in that component).

### Wasm boundary crossings

At T=100 per forward pass: approximately **7,307 JS→Wasm calls** (formula: `T×9 + nh + 1` per layer × 8 layers + 3 final). Each crossing is a function-call-level overhead in V8 (roughly 10–50 ns each). At 7,300 calls this is ~73–365 μs of pure overhead per step — 1–5% of a 10 ms step, not dominant today. A KV cache reduces this to ~50 calls per incremental step (the overhead becomes negligible).

---

## 3. Candidate evaluation

### Candidate 1: Attention inner loops → `attention_f32` Wasm export

**Current location:** Pure JS in `forward()`, lines 138–155.  
The Q·Kᵀ dot products (lines 139–146) and weighted V sum (lines 149–155) are nested JS loops. The causal softmax between them is already a Wasm call (`softmax_causal_f32`).

**Difficulty:** Moderate.  
A new `attention_f32` function takes pointers to the QKV buffer, scores buffer, and attn output buffer, plus seq/d/nh/dh dimensions. It replaces roughly 12 lines of JS with one Wasm call covering all heads. The tricky part is the strided memory layout: QKV is stored interleaved as `[Q₀|K₀|V₀, Q₁|K₁|V₁, ...]` (3×d per position), so the Rust code must stride correctly when reading per-head slices.

**Expected speedup:** At T=255, eliminates ~45% of estimated wall-clock on the attention component. Total step speedup: roughly 1.05× at T=10, 1.35× at T=100, 1.8× at T=255 (before KV cache). After KV cache is added, this remains valuable for the single-new-query attention dot products which are still JS today.

**Blocked by:** Nothing — can be done independently.

**Recommendation:** **Move. Priority 2.**

---

### Candidate 2: Embedding lookup → `embed_f32` Wasm export

**Current location:** JS, `forward()` lines 110–116. Two nested loops: for each position p, copy `teW[tid*d+j] + peW[p*d+j]` into the embedding buffer.

**Difficulty:** Easy. The operation is a simple gather-and-add: `emb[p*d+j] = token_embed[tokens[p]*d+j] + pos_embed[p*d+j]`.

**Expected speedup:** Negligible. The embedding lookup is O(T×d). At T=100, d=768, that is 76,800 additions — perhaps 50–100 μs in JS, far less than 1% of total step time.

**Blocked by:** Nothing.

**Recommendation:** **Defer.** Implement only as part of a "eliminate all JS arithmetic" polish pass, or when adding SIMD to the kernel. Not worth standalone effort.

---

### Candidate 3: Batch matmul — per-position matvec → single matmat call

**Current location:** The QKV projection loop (lines 122–130), O-projection loop (lines 158–160), and FFN loops (lines 163–173) each call `matmul_f32w` once per position. At T=100 that is 100 separate matvec calls per projection per layer.

**Difficulty:** Moderate (new Wasm export).  
The existing `matmul_f32w` signature processes one vector at a time. A new `matmul_batch_f32w` export would handle T rows in a single call, reducing boundary crossings from T×6 per layer to 6 per layer and improving cache utilization (the weight matrix is read once per batch rather than once per position).

**Expected speedup from batching alone:** 5–20% on the Wasm matmul component for the prefill phase (weight matrix for QKV at d=768 is 2.25 MB — still fits in L3 on most CPUs but batching amortizes the load cost). After KV cache, only the prefill phase processes T positions at once; incremental steps are already a single row.

**Blocked by:** Nothing in isolation. Most useful to implement alongside KV cache as the "prefill" fast path.

**Recommendation:** **Defer until KV cache, then include.** This is low-complexity Rust (the inner loop is just the existing `matmul_f32w` inner loop with a batch dimension) and natural to add in the same PR as KV cache.

---

### Candidate 4: KV cache

**Current location:** Does not exist. `generate()` calls `forward()` with the full growing token sequence on every step.

**Difficulty:** Hard — restructures both the Wasm memory model and the TS orchestrator.

**Expected speedup:** By far the largest single win. Eliminates O(T²) recompute: at T=100, step time drops ~100× on QKV projections and ~150× on JS attention loops. This is the difference between a multi-second response and a sub-100 ms step at full context.

**Design: where do K/V buffers live?**

**Option A — K/V in Wasm memory (recommended):**  
Allocate a persistent KV cache region in Wasm memory beyond the weight section. Per layer: two buffers of shape `[max_len × d]` (K and V). For Specter: 8 layers × 2 × 256 × 768 × 4 bytes = 12.6 MB — fits comfortably alongside the 219 MB weight section. Wasm owns the buffers; the TS orchestrator passes the current position index on each incremental step. Zero JS↔Wasm copying of large K/V arrays.

**Option B — K/V in JS `Float32Array`:**  
TS allocates typed arrays and copies new K/V slices to Wasm on each step. Simpler to implement, but copies ~6 MB per step at Specter scale (8 layers × 2 × 256 × 768 × 4 bytes). Not recommended.

**Blocked by:** Nothing external. Add `matmul_batch_f32w` in the same PR for efficient prefill.

**Recommendation:** **Move. Priority 1.** This is the single most impactful change and a prerequisite for Specter being usable in the browser. Without a KV cache, at T=200 the in-browser forward pass recomputes 11 billion Wasm MACs plus 369 million JS attention MACs from scratch every step.

---

### Candidate 5: Sampling loop

**Current location:** `generate()`, lines 212–220. Softmax with temperature, then multinomial draw over vocab_size=258.

**Recommendation:** **Keep in JS.** The sampling loop operates on 258 floats — trivially fast. Temperature, top-k, and top-p controls belong at the JS level where they can be adjusted without recompiling Wasm.

---

### Candidate 6: Dead 4-bit packed path (`matmul_f32`, `matmul_no_bias_f32`)

**Current location:** `wasm/src/lib.rs` lines 25–77. Two Rust functions exist and compile but are never called from TypeScript.

**What exists:**
- `matmul_f32`: packed 4-bit weights (two nibbles per byte) with float32 biases and input/output
- `matmul_no_bias_f32`: same without bias
- Weight unpacking: `(raw_nibble - 8) * GLOBAL_WEIGHT_SCALE` where `GLOBAL_WEIGHT_SCALE = 0.4` (compile-time constant)
- Representable values: {−3.2, −2.8, ..., −0.4, 0.0, 0.4, ..., +2.8} — 16 levels, step 0.4

**What is missing to wire it up:**

1. **Python quantizer.** The serializer produces float32 `.bin` files. There is no code to quantize float32 weights to the nibble-packed format. A quantizer must: clip weights to the representable range, round to the nearest value, and pack two weights per byte matching the `unpack_weight_f32` bit pattern.

2. **WasmApi interface.** The TypeScript `WasmApi` interface (lines 8–16 of `tier2_transformer.ts`) does not include `matmul_f32` or `matmul_no_bias_f32`. Both signatures must be added.

3. **forward() dispatch.** The orchestrator needs to know at runtime whether a float32 or 4-bit model was loaded. Cleanest approach: a `quantization` field in the JSON manifest (`"float32"` or `"int4_nibble"`), branching on it in `forward()` to call `matmul_f32` vs `matmul_f32w`.

4. **Scale calibration.** `GLOBAL_WEIGHT_SCALE = 0.4` is baked into the compiled Wasm binary. If actual trained weights have a wider distribution (e.g., attention projection weights with magnitude >3.2), they clip to the boundary causing significant quality loss. Options: (a) store a per-tensor scale in the manifest and add a runtime-parameterized unpack in Rust; (b) use quantization-aware training (QAT) to constrain weights to the fixed grid. Option (a) is recommended.

5. **Separate model binary.** A 4-bit Specter `.bin` is ~28 MB vs 219 MB float32 — this is the shipping blocker for Specter (GitHub's 100 MB per-file hard limit). The serializer must produce a separate 4-bit `.bin` paired with a manifest that signals `"quantization": "int4_nibble"`.

**Correctness note:** The `matmul_f32` and `matmul_no_bias_f32` implementations are algorithmically correct — they unpack each weight inline during the multiply-accumulate. The inner loop is not SIMD-vectorized (nibble unpacking prevents straightforward auto-vectorization), so 4-bit in Wasm will be slower per FLOP than float32 `matmul_f32w`, but the 8× memory reduction means far better cache utilization for large weight matrices, likely netting a throughput win at Specter scale.

**Recommendation:** **Defer activation, do not remove the Rust code.** The 4-bit path is a shipping prerequisite for Specter (219 MB float32 cannot be committed to git). However, activating it requires a Python quantizer and TS dispatch changes, and quality cannot be validated without a trained Specter checkpoint.

---

## 4. Priority order and recommendation table

| # | Candidate | Current | Recommendation | Est. speedup | Dependencies |
|---|---|---|---|---|---|
| 1 | **KV cache** | Absent | **Move — Priority 1** | 100–300× per step at T>50 | None (add `matmul_batch_f32w` in same PR for prefill) |
| 2 | **Attention inner loops** (Q·Kᵀ + wV) | JS | **Move — Priority 2** | 1.35–1.8× step at T≥100 | None (independent) |
| 3 | **Batch matmul** (prefill phase) | TS per-position calls | **Move — Priority 3** | 5–20% on prefill | Best implemented alongside KV cache |
| 4 | **4-bit quantization path** | Wasm (dead) | **Activate — Priority 4** | 8× memory; ~1.2× throughput | Python quantizer; trained Specter checkpoint; manifest flag |
| 5 | **Sampling loop** | TS | **Keep in JS** | N/A | — |
| 6 | **Embedding lookup** | JS | **Defer** | <1% | — |

**What gives the most bang before Specter ships:**

The answer is unambiguous: **KV cache first**. Without it, Specter at 256 tokens will be multiple seconds per token in the browser regardless of any other optimization. After KV cache, **attention-in-Wasm** eliminates the remaining JS interpreter bottleneck at long contexts. Together these two changes make Specter's per-step cost nearly constant in T and fully compiled.

The **4-bit path** is not a performance optimization — it is a **shipping requirement** for Specter. The 219 MB float32 binary cannot be committed to the repository (GitHub's 100 MB hard limit). Without quantization, Specter cannot ship at all.

---

## 5. API sketches for high-priority moves

### 5.1 KV cache — new Wasm export + incremental forward

**Rust addition (`wasm/src/lib.rs`):**

```rust
/// Single-position attention with KV cache.
/// Computes Q·K scores for the new position against all cached positions,
/// softmaxes, and accumulates weighted V into attn_out.
/// Also writes k_new/v_new into the cache at index `step`.
///
/// k_cache / v_cache: [max_len × d] — persistent cache for this layer.
/// q:       [d] — Q for the new position.
/// k_new:   [d] — K for the new position.
/// v_new:   [d] — V for the new position.
/// attn_out:[d] — output written here (caller zeroes before first layer).
#[no_mangle]
pub unsafe extern "C" fn attn_cached_f32(
    q: *const f32,
    k_new: *const f32,
    v_new: *const f32,
    k_cache: *mut f32,    // [max_len × d]
    v_cache: *mut f32,    // [max_len × d]
    attn_out: *mut f32,   // [d]
    step: i32,            // current position (0-indexed); cache has 0..step-1
    max_len: i32,
    d: i32,
    n_heads: i32,
    d_head: i32,
)
```

**TypeScript additions (`tier2_transformer.ts`):**

```typescript
// Extended WasmApi interface:
interface WasmApiWithCache extends WasmApi {
  attn_cached_f32(
    q: number, k_new: number, v_new: number,
    k_cache: number, v_cache: number, attn_out: number,
    step: number, max_len: number, d: number,
    n_heads: number, d_head: number
  ): void;
}

// KV cache memory size:
// n_layers * 2 * max_len * d_model * 4 bytes
// For Specter: 8 * 2 * 256 * 768 * 4 = 12,582,912 bytes (~12 MB)
// Lives in Wasm memory beyond the weight section and forward scratch.

// KV cache reset (call when starting a new prompt):
export function resetKVCache(api: WasmApi, kvBase: number, arch: Arch): void {
  const floats = arch.n_layers * 2 * arch.max_len * arch.d_model;
  new Float32Array(api.memory.buffer, kvBase, floats).fill(0);
}

// Incremental forward (one token, O(T) per step):
export function forwardIncremental(
  api: WasmApiWithCache,
  sec: Record<string, SectionDef>,
  arch: Arch,
  base: number,
  kvBase: number,      // pointer to KV cache region in Wasm memory
  newToken: number,    // token id for the new position
  step: number,        // current position index (0-indexed)
): Float32Array        // logits[vocab_size]
```

The full-sequence `forward()` is preserved for the prefill phase (processing the initial prompt in one shot before switching to incremental steps).

### 5.2 Attention inner loops → `attention_f32` Wasm export

Covers the full-sequence (prefill) case and is independent of the KV cache.

**Rust addition (`wasm/src/lib.rs`):**

```rust
/// Full-sequence multi-head self-attention (causal).
/// Replaces the JS Q·Kᵀ and weighted-V-sum loops in forward().
///
/// qkv layout: [seq × (3 × d)] f32, interleaved Q/K/V per position.
///   Q for position t, head h, dim x: qkv[t*(3*d) + h*dh + x]
///   K for position t, head h, dim x: qkv[t*(3*d) + d + h*dh + x]
///   V for position t, head h, dim x: qkv[t*(3*d) + 2*d + h*dh + x]
///
/// scores: scratch [seq × seq] f32 (caller-allocated, reused across heads).
/// out:    [seq × d] f32 — zeroed by caller before this call.
#[no_mangle]
pub unsafe extern "C" fn attention_f32(
    qkv: *const f32,
    scores: *mut f32,
    out: *mut f32,
    seq: i32,
    d: i32,
    n_heads: i32,
    d_head: i32,
)
```

**TypeScript call site (replaces lines 138–155 of `tier2_transformer.ts`):**

```typescript
// Add to WasmApi interface:
// attention_f32(qkv: number, scores: number, out: number,
//               seq: number, d: number, n_heads: number, d_head: number): void;

// In forward(), replace the JS attention block with:
attn.fill(0);
api.attention_f32(qOff, sOff, aOff, seq, d, nh, dh);
// The separate softmax_causal_f32 call is absorbed into attention_f32.
```

### 5.3 Batch matmul → `matmul_batch_f32w`

```rust
/// Matrix-matrix multiply: input [batch × in_dim] × weights [out_dim × in_dim]ᵀ
/// + broadcast bias [out_dim] → output [batch × out_dim].
/// Drop-in batch replacement for T separate matmul_f32w calls.
#[no_mangle]
pub unsafe extern "C" fn matmul_batch_f32w(
    weights: *const f32,  // [out_dim × in_dim]
    biases: *const f32,   // [out_dim]
    input: *const f32,    // [batch × in_dim]
    output: *mut f32,     // [batch × out_dim]
    in_dim: i32,
    out_dim: i32,
    batch: i32,
)
```

In `forward()`, the QKV projection loop (currently `T × 3` separate `matmul_f32w` calls per layer) becomes 3 calls to `matmul_batch_f32w` with `batch=T`. Similarly for O-projection (1 call) and FFN (2 calls). Total per-layer calls drop from `T×9 + nh + 1` to `9 + nh`.

---

## 6. Dead 4-bit path: full activation checklist

The Rust code in `matmul_f32` and `matmul_no_bias_f32` is algorithmically correct but cannot be used until:

| # | Missing component | Where | Notes |
|---|---|---|---|
| 1 | Python quantizer | `py/serialize_4bit.py` | Clip to [−3.2, 2.8], round to nearest representable value, nibble-pack; output `.4bit.bin` |
| 2 | Manifest field | JSON serializer | Add `"quantization": "int4_nibble"` so the TS orchestrator can dispatch |
| 3 | WasmApi TS interface | `tier2_transformer.ts` lines 8–16 | Add `matmul_f32(w,b,inp,out,inD,outD)` and `matmul_no_bias_f32(w,inp,out,inD,outD)` |
| 4 | forward() dispatch | `forward()` function | Branch on `manifest.quantization` to call `matmul_f32` vs `matmul_f32w` |
| 5 | Weight scale calibration | `wasm/src/lib.rs` | `GLOBAL_WEIGHT_SCALE = 0.4` is compile-time; per-tensor scale stored in manifest is strongly recommended |
| 6 | Section size adjustment | `instantiateModel` | 4-bit weight sections are half the byte length of float32; scratch buffers stay float32 |
| 7 | Quality validation | `test/parity_4bit.test.js` | Compare 4-bit vs float32 logits; expect divergence ~0.05–0.2 (ok); >1.0 means broken scale |

**Critical: the `GLOBAL_WEIGHT_SCALE = 0.4` problem.** This constant is baked into the compiled Wasm binary. If actual trained weights have a wider distribution (e.g., attention projection weights with magnitude >3.2), they clip to the boundary, causing significant quality loss. The quantizer must measure the per-tensor weight range and store a per-tensor scale in the manifest, requiring a runtime-parameterized unpack function in Rust rather than the current compile-time constant.

**Memory geometry for 4-bit Specter:**

| Component | Float32 | 4-bit packed | Notes |
|---|---|---|---|
| QKV+O per layer (4 × d×d) | 9.0 MB | 1.1 MB | 8× reduction |
| FFN per layer (2 × d×d_ff) | 18.0 MB | 2.25 MB | 8× reduction |
| Embeddings (token + pos embed) | 1.5 MB | 0.19 MB | 8× reduction |
| Biases (stay float32) | ~0.3 MB | ~0.3 MB | No change |
| **Total Specter** | **~219 MB** | **~28 MB** | Fits under GitHub 100 MB limit |

---

## 7. Action order before Specter ships

1. **KV cache** (1–2 weeks): gating item; Specter is unusable without it. Implement `attn_cached_f32` in Rust, add `forwardIncremental` in TS, refactor `generate()` to prefill + incremental. Include `matmul_batch_f32w` in the same PR for prefill efficiency.

2. **`attention_f32` Wasm export** (2 days): eliminates the only remaining JS interpreter hot path. ~30 lines of Rust, 5 lines of TS change.

3. **4-bit serializer + path activation** (1 week): shipping requirement for Specter; the Rust side already exists. Work is the Python quantizer and TS dispatch. Validate with a parity test against a trained checkpoint.

4. **`matmul_batch_f32w` for prefill** (1 day): low-complexity addition; natural to bundle with KV cache in the same PR.
