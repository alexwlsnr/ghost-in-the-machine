# WebGPU Inference Plan

## Why it's needed

For Specter (57M params, d=768, ctx=256), CPU Wasm inference is ~1-3 seconds per
token — unusable. Each forward step requires 8 layers × 6 matmuls where each matrix
is [768, 768] or [768, 3072]. On GPU, all 3072 output neurons compute in parallel;
realistic estimate is **20-50ms per token** (20-50 tok/s).

Spec512 (25.6M, d=512) is borderline on CPU — maybe 2-5 tok/s with SIMD + KV cache.
Specter needs WebGPU to be usable at all.

## Architecture — it fits our existing stack

The current split is already right:
- **TS orchestrator** — control plane: forward pass logic, buffer layout, sampling
- **Wasm kernel** — compute plane: matmuls, softmax, layer_norm

For WebGPU, keep the orchestrator; replace the Wasm dispatch with GPU dispatch.
The `makeMatmulDispatch()` factory already abstracts this — add a GPU variant.

Detection on load:
```typescript
const gpu = await navigator.gpu?.requestAdapter();
const useGPU = !!gpu && modelSize > GPU_THRESHOLD_BYTES;
```

Route: Wisp + Shade → CPU Wasm (fast enough). Spec512 + Specter → WebGPU if
available, Wasm fallback if not.

## What needs to be built

### 1. WGSL compute shaders (~300 lines)

Four shaders covering the hot path:

**matmul.wgsl** — replaces `matmul_f32w`. One GPU thread per output element.
```wgsl
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out = gid.x;
    var sum = biases[out];
    for (var i = 0u; i < in_dim; i++) {
        sum += weights[out * in_dim + i] * input[i];
    }
    output[out] = sum;
}
```

**attention.wgsl** — replaces the JS attention loops in `forwardIncremental`.
One thread group per head. Q·Kᵀ + causal softmax + weighted V in one dispatch.

**layernorm.wgsl** — two-pass reduction (mean, then variance) using workgroup
shared memory.

**softmax.wgsl** — numerically stable max-then-exp, parallel over vocab.

4-bit quantized path: unpack nibbles inside the shader — same convention as the
Rust kernel, all 3072 outputs unpack in parallel on GPU cores.

### 2. Buffer management in TS (~400 lines)

```typescript
class GPUModel {
  weightBuffers: Map<string, GPUBuffer>;  // model weights on VRAM — uploaded once
  activationBuffers: GPUBuffer[];          // reused each forward step
  kvCache: GPUBuffer[];                    // stays on GPU across tokens

  async upload(model: LoadedModel): Promise<void> { /* createBuffer + writeBuffer */ }
  async forward(tokens: number[]): Promise<Float32Array> { /* dispatch shaders */ }
}
```

Key: only read back logits (~258 floats) to CPU for sampling. Everything else
stays on GPU.

### 3. Orchestrator extension (~200 lines)

Add GPU variant of `makeMatmulDispatch` and `forwardIncremental`. The existing
code structure makes this clean — a `createGPUModel()` function alongside
`instantiateModel()`, same output shape.

## Effort estimate

| Phase | Time |
|---|---|
| WGSL shaders (matmul, attention, layernorm, softmax) | 3-5 days |
| Buffer management and upload pipeline | 3-5 days |
| Orchestrator GPU path + feature detection | 2-3 days |
| Debugging (GPU is hard — invisible errors) | 1-2 weeks |
| **Total** | **4-6 weeks** |

## Shortcut: onnxruntime-web

Export our trained model to ONNX (1 day), use `onnxruntime-web` with WebGPU
backend. Reduces to ~1 week total. Trade-offs:
- +5MB JS dependency
- Loses custom 4-bit quantization (onnxruntime uses its own quant)
- No custom KV cache
- Less control over inference

Right choice for a quick prototype; native WGSL for production.

## Browser support

| Browser | WebGPU | Notes |
|---|---|---|
| Chrome 113+ | ✅ | Desktop + Android |
| Firefox | ⚠️ | Behind flag, not ready |
| Safari 17.2+ | ✅ | macOS/iOS partial |
| Samsung Internet | ✅ | Recent versions |

~70% of users get GPU path; 30% fall back to Wasm CPU (fine for Wisp/Shade).

## Sequencing with current work

1. ✅ Wasm SIMD (done) — necessary foundation, also benefits Spec512 on CPU
2. ✅ KV cache (done) — required for any model at ctx=256 to be usable
3. → Spec512 training + Wasm deployment (immediate)
4. → Specter training + 4-bit serialization
5. → WebGPU path for Spec512/Specter (after models are validated on CPU)

Don't build WebGPU until the models are trained and quality-validated. The CPU
Wasm path is the development/validation environment; WebGPU is the production
delivery for large models.
