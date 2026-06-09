/**
 * WebGPU inference engine for Ghost in the Machine.
 *
 * Replaces the Wasm matmul dispatch for large models (>= GPU_THRESHOLD_BYTES).
 * Supports classic arch: LayerNorm, ReLU FFN, learned positional embeddings.
 * Weight formats: float32, int8 (per-tensor scale), int4g (per-group 4-bit).
 *
 * All weights are uploaded to GPU VRAM once at construction time.
 * The KV cache lives on GPU across tokens; only 258 logit floats are
 * transferred back to CPU per generated token.
 *
 * Usage:
 *   const gpu = await GPUEngine.create(arch, sec, wasmMemory, base);
 *   if (gpu) {
 *     const logits = await gpu.prefill(promptTokens);
 *     const next   = await gpu.step(sampledToken);
 *     gpu.reset();
 *     gpu.destroy();
 *   }
 */
/**
 * Models whose estimated fp32 matrix weight size (bytes) exceeds this get the
 * GPU path. Set between Shade (~40 MB) and Spec512 (~95 MB) so Wisp and Shade
 * stay on Wasm (they're already fast) and Spec512+ gets GPU uplift.
 */
export const GPU_THRESHOLD_BYTES = 60000000;
/** True when the model architecture is large enough to benefit from GPU. */
export function shouldUseGPU(arch) {
    // Estimate fp32 byte size of attention + FFN matrices only (bulk of params)
    const matrixParams = arch.n_layers * (4 * arch.d_model ** 2 + 2 * arch.d_model * arch.d_ff);
    return matrixParams * 4 >= GPU_THRESHOLD_BYTES;
}
/** Check whether WebGPU is available in this environment. */
export async function isWebGPUAvailable() {
    try {
        return !!(await navigator.gpu?.requestAdapter());
    }
    catch {
        return false;
    }
}
// ── WGSL compute shaders ────────────────────────────────────────────────────
const EMBED_SHADER = /* wgsl */ `
struct P { token_id: u32, pos: u32, d: u32 }
@group(0) @binding(0) var<storage, read>       tok:   array<f32>;
@group(0) @binding(1) var<storage, read>       pos_e: array<f32>;
@group(0) @binding(2) var<storage, read_write> x:     array<f32>;
@group(0) @binding(3) var<uniform>             p:     P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j < p.d) { x[j] = tok[p.token_id * p.d + j] + pos_e[p.pos * p.d + j]; }
}`;
const LAYERNORM_SHADER = /* wgsl */ `
// LayerNorm via parallel reduction (workgroup_size=256, handles d up to 512).
// Each thread owns elements at indices [lid, lid+256, lid+512, ...].
struct P { d: u32 }
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform>             p: P;
var<workgroup> tmp: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let d = p.d;
  var s = 0.0;
  for (var i = t; i < d; i += 256u) { s += x[i]; }
  tmp[t] = s; workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (t < st) { tmp[t] += tmp[t + st]; }
    workgroupBarrier();
  }
  let mean = tmp[0] / f32(d);
  var sq = 0.0;
  for (var i = t; i < d; i += 256u) { let v = x[i] - mean; sq += v * v; }
  tmp[t] = sq; workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (t < st) { tmp[t] += tmp[t + st]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(tmp[0] / f32(d) + 1e-5f);
  for (var i = t; i < d; i += 256u) { y[i] = (x[i] - mean) * inv * w[i] + b[i]; }
}`;
const MATMUL_F32_SHADER = /* wgsl */ `
// y = W @ x + bias.  W is [out_dim, in_dim].  One thread per output element.
struct P { in_dim: u32, out_dim: u32, has_bias: u32 }
@group(0) @binding(0) var<storage, read>       w:    array<f32>;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:    array<f32>;
@group(0) @binding(4) var<uniform>             p:    P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= p.out_dim) { return; }
  var sum = 0.0;
  let base = row * p.in_dim;
  for (var col = 0u; col < p.in_dim; col++) { sum += w[base + col] * x[col]; }
  if (p.has_bias != 0u) { sum += bias[row]; }
  y[row] = sum;
}`;
const MATMUL_INT4G_SHADER = /* wgsl */ `
// 4-bit grouped-quantized matmul.
// Nibble packing convention (matches py/serialize.py):
//   byte[b] = (hi << 4) | lo
//   hi = weight at flat_idx 2*b   (even) — stored as (val+8) & 0xF
//   lo = weight at flat_idx 2*b+1 (odd)  — stored as (val+8) & 0xF
//   val in -8..7.  Dequantize: float_val = nibble - 8.
// Per-group scale: scales[row * n_groups + group].
// Buffer binding as array<u32> gives little-endian byte access.
struct P { in_dim: u32, out_dim: u32, group_size: u32, has_bias: u32 }
@group(0) @binding(0) var<storage, read>       nib:  array<u32>;
@group(0) @binding(1) var<storage, read>       scl:  array<f32>;
@group(0) @binding(2) var<storage, read>       x:    array<f32>;
@group(0) @binding(3) var<storage, read>       bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:    array<f32>;
@group(0) @binding(5) var<uniform>             p:    P;
fn w_val(flat: u32) -> f32 {
  let byte_idx = flat >> 1u;
  let byte = (nib[byte_idx >> 2u] >> ((byte_idx & 3u) << 3u)) & 0xFFu;
  let nibble = select(byte & 0xFu, (byte >> 4u) & 0xFu, (flat & 1u) == 0u);
  return f32(i32(nibble) - 8);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= p.out_dim) { return; }
  let n_groups = (p.in_dim + p.group_size - 1u) / p.group_size;
  var sum = 0.0;
  for (var g = 0u; g < n_groups; g++) {
    let gs = g * p.group_size;
    let ge = min(gs + p.group_size, p.in_dim);
    let sc = scl[row * n_groups + g];
    for (var col = gs; col < ge; col++) {
      sum += w_val(row * p.in_dim + col) * sc * x[col];
    }
  }
  if (p.has_bias != 0u) { sum += bias[row]; }
  y[row] = sum;
}`;
const WRITE_KV_SHADER = /* wgsl */ `
// Copy d floats from src into dst at row = pos.
struct P { pos: u32, d: u32 }
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             p:   P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < p.d) { dst[p.pos * p.d + i] = src[i]; }
}`;
const ATTENTION_SHADER = /* wgsl */ `
// Incremental attention for one query token vs. all cached K/V.
// Dispatch: n_heads workgroups, workgroup_size=64 (= d_head for Spec512).
// Scores stored in workgroup shared memory (1024 floats = max_ctx).
struct P { seq_len: u32, n_heads: u32, d_head: u32, d_model: u32 }
@group(0) @binding(0) var<storage, read>       q:  array<f32>;
@group(0) @binding(1) var<storage, read>       kc: array<f32>;
@group(0) @binding(2) var<storage, read>       vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> o:  array<f32>;
@group(0) @binding(4) var<uniform>             p:  P;
var<workgroup> sc: array<f32, 1024>;
@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id)        wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let head  = wid.x;
  let t     = lid.x;
  let h_off = head * p.d_head;
  let scale = 1.0 / sqrt(f32(p.d_head));
  let d     = p.d_model;
  // QK^T scores — threads split the seq_len dimension
  for (var kj = t; kj < p.seq_len; kj += 64u) {
    var dot = 0.0;
    for (var xi = 0u; xi < p.d_head; xi++) {
      dot += q[h_off + xi] * kc[kj * d + h_off + xi];
    }
    sc[kj] = dot * scale;
  }
  workgroupBarrier();
  // Softmax: thread 0 handles it sequentially (seq_len is small)
  if (t == 0u) {
    var mx = -1e30f;
    for (var kj = 0u; kj < p.seq_len; kj++) { mx = max(mx, sc[kj]); }
    var sm = 0.0;
    for (var kj = 0u; kj < p.seq_len; kj++) { sc[kj] = exp(sc[kj] - mx); sm += sc[kj]; }
    let iv = 1.0 / sm;
    for (var kj = 0u; kj < p.seq_len; kj++) { sc[kj] *= iv; }
  }
  workgroupBarrier();
  // Weighted V sum — threads split the d_head dimension
  for (var xi = t; xi < p.d_head; xi += 64u) {
    var val = 0.0;
    for (var kj = 0u; kj < p.seq_len; kj++) { val += sc[kj] * vc[kj * d + h_off + xi]; }
    o[h_off + xi] = val;
  }
}`;
const ADD_SHADER = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i < p.n) { a[i] += b[i]; }
}`;
const RELU_SHADER = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform>             p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i < p.n) { x[i] = max(0.0f, x[i]); }
}`;
// int8 weights packed as bytes in u32 (4 weights per u32, little-endian byte order).
// scale is per-tensor, stored as f32 in the uniform (at byte offset 8 in the struct).
const MATMUL_INT8_SHADER = /* wgsl */ `
struct P { in_dim: u32, out_dim: u32, scale: f32, has_bias: u32 }
@group(0) @binding(0) var<storage, read>       w:    array<u32>;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:    array<f32>;
@group(0) @binding(4) var<uniform>             p:    P;
fn i8_val(flat: u32) -> f32 {
  let byte_u = (w[flat >> 2u] >> ((flat & 3u) << 3u)) & 0xFFu;
  return f32(i32(byte_u) - select(0, 256, byte_u >= 128u));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= p.out_dim) { return; }
  var sum = 0.0;
  let base = row * p.in_dim;
  for (var col = 0u; col < p.in_dim; col++) { sum += i8_val(base + col) * p.scale * x[col]; }
  if (p.has_bias != 0u) { sum += bias[row]; }
  y[row] = sum;
}`;
// ── Buffer utilities ─────────────────────────────────────────────────────────
// Use raw WebGPU spec values rather than GPUBufferUsage.* globals — the globals
// may not exist at module evaluation time in non-WebGPU environments.
// Values from https://gpuweb.github.io/gpuweb/#buffer-usage
const BU_COPY_SRC = 0x04;
const BU_COPY_DST = 0x08;
const BU_UNIFORM = 0x40;
const BU_STORAGE = 0x80;
const BU_MAP_READ = 0x01;
const ST = BU_STORAGE | BU_COPY_DST; // storage buffer (written from GPU, may copy out)
const UN = BU_UNIFORM | BU_COPY_DST; // uniform buffer (written from CPU each token)
const GPU_MAP_READ = 0x01; // GPUMapMode.READ
function uploadBuf(device, data, usage) {
    const buf = device.createBuffer({ size: (data.byteLength + 3) & ~3, usage, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buf.unmap();
    return buf;
}
function uniformU32(device, values) {
    // Chrome requires uniform buffers to be >= 16 bytes regardless of struct size.
    // Pad to 4 u32 values minimum; extra bytes are harmless (not read by the shader).
    const arr = new Uint32Array(Math.max(4, values.length));
    arr.set(values);
    return uploadBuf(device, arr, UN);
}
function emptyStorageBuf(device, byteSize, extraUsage = 0) {
    return device.createBuffer({ size: (byteSize + 3) & ~3, usage: ST | extraUsage });
}
// ── GPUEngine ────────────────────────────────────────────────────────────────
export class GPUEngine {
    get seqLen() { return this._seqLen; }
    constructor(device, arch, pip) {
        // Per-layer norm weights
        this.ln1w = [];
        this.ln1b = [];
        this.ln2w = [];
        this.ln2b = [];
        // Per-layer attention + FFN weight tensors
        this.qT = [];
        this.kT = [];
        this.vT = [];
        this.oT = [];
        this.ff1T = [];
        this.ff2T = [];
        // KV cache (one pair per layer, persists across tokens)
        this.kCache = [];
        this.vCache = [];
        this._seqLen = 0;
        this.device = device;
        this.arch = arch;
        this.pip = pip;
    }
    /**
     * Build a GPUEngine from a loaded model's binary data.
     * Returns null if WebGPU is unavailable or the model architecture
     * is not supported (non-classic or uses RoPE/SwiGLU).
     */
    static async create(arch, sec, wasmMem, base) {
        // Classic arch only: no RoPE, no SwiGLU, no RMSNorm
        if (arch.use_rope || arch.use_swiglu || arch.use_rmsnorm)
            return null;
        const gpu = navigator.gpu;
        if (!gpu)
            return null;
        const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter)
            return null;
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            },
        });
        const pip = GPUEngine._compilePipelines(device);
        const engine = new GPUEngine(device, arch, pip);
        engine._uploadWeights(arch, sec, wasmMem, base);
        engine._allocActivations(arch);
        return engine;
    }
    static _compilePipelines(device) {
        const mk = (src) => device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: src }), entryPoint: 'main' },
        });
        return {
            embed: mk(EMBED_SHADER),
            ln: mk(LAYERNORM_SHADER),
            matF32: mk(MATMUL_F32_SHADER),
            matI4g: mk(MATMUL_INT4G_SHADER),
            matI8: mk(MATMUL_INT8_SHADER),
            writeKV: mk(WRITE_KV_SHADER),
            attn: mk(ATTENTION_SHADER),
            add: mk(ADD_SHADER),
            relu: mk(RELU_SHADER),
        };
    }
    _uploadWeights(arch, sec, mem, base) {
        const { device } = this;
        const { d_model: d, d_ff: ff, n_layers: nl } = arch;
        const mem8 = new Uint8Array(mem.buffer);
        const gs = 32; // int4g group_size
        const f32Sec = (name) => uploadBuf(device, new Float32Array(mem8.buffer, base + sec[name].offset, sec[name].size / 4), ST);
        const mkWT = (wName, bName) => {
            const s = sec[wName];
            const [out, inp] = s.shape;
            const wt = { biasBuf: f32Sec(bName), dtype: s.dtype, inDim: inp, outDim: out };
            if (s.dtype === 'int4g') {
                const nibBytes = new Uint8Array(mem8.buffer, base + s.offset, s.size);
                wt.nibBuf = uploadBuf(device, nibBytes, ST);
                wt.sclBuf = uploadBuf(device, new Float32Array(mem8.buffer, base + s.scales_offset, s.scales_size / 4), ST);
            }
            else if (s.dtype === 'int8') {
                wt.int8Buf = uploadBuf(device, new Uint8Array(mem8.buffer, base + s.offset, s.size), ST);
                // Pack {in_dim: u32, out_dim: u32, scale: f32, has_bias: u32} into 16-byte uniform
                const pb = new ArrayBuffer(16);
                new Uint32Array(pb)[0] = inp;
                new Uint32Array(pb)[1] = out;
                new Float32Array(pb)[2] = s.scale ?? 1.0;
                new Uint32Array(pb)[3] = 1;
                wt.paramBuf = uploadBuf(device, new Uint8Array(pb), UN);
            }
            else {
                wt.f32Buf = f32Sec(wName);
            }
            return wt;
        };
        // Embeddings
        this.tokEmb = f32Sec('token_embed');
        this.posEmb = f32Sec('pos_embed');
        // Final norm + head
        this.lnFw = f32Sec('lnf_w');
        this.lnFb = f32Sec('lnf_b');
        this.headW = f32Sec('head_weight');
        // Per-layer
        for (let li = 0; li < nl; li++) {
            const p = `enc${li}`;
            this.ln1w.push(f32Sec(`${p}_ln1_w`));
            this.ln1b.push(f32Sec(`${p}_ln1_b`));
            this.ln2w.push(f32Sec(`${p}_ln2_w`));
            this.ln2b.push(f32Sec(`${p}_ln2_b`));
            this.qT.push(mkWT(`${p}_q_weight`, `${p}_q_bias`));
            this.kT.push(mkWT(`${p}_k_weight`, `${p}_k_bias`));
            this.vT.push(mkWT(`${p}_v_weight`, `${p}_v_bias`));
            this.oT.push(mkWT(`${p}_o_weight`, `${p}_o_bias`));
            this.ff1T.push(mkWT(`${p}_ff1_weight`, `${p}_ff1_bias`));
            this.ff2T.push(mkWT(`${p}_ff2_weight`, `${p}_ff2_bias`));
        }
        // Fixed uniform buffers — written once
        this.lnPBuf = uniformU32(device, [d]);
        this.addPBuf = uniformU32(device, [d]);
        this.reluPBuf = uniformU32(device, [ff]);
        // Shader struct P = { in_dim, out_dim, group_size, has_bias }
        // FF1 weight is [d_ff, d] (out=d_ff, in=d); FF2 weight is [d, d_ff] (out=d, in=d_ff)
        this.matQKVOPBuf = uniformU32(device, [d, d, gs, 1]); // in=d,  out=d
        this.matFF1PBuf = uniformU32(device, [d, ff, gs, 1]); // in=d,  out=d_ff
        this.matFF2PBuf = uniformU32(device, [ff, d, gs, 1]); // in=d_ff, out=d
        this.headPBuf = uniformU32(device, [d, arch.vocab_size, 0]);
        // Zero bias stub (head projection has no bias; we pass this to satisfy the binding)
        this.zeroBuf = device.createBuffer({ size: 4, usage: BU_STORAGE });
        // Per-token uniform buffers — placeholder values, updated each step
        this.embedPBuf = uniformU32(device, [0, 0, d]);
        this.attnPBuf = uniformU32(device, [1, arch.n_heads, d / arch.n_heads, d]);
        this.wkvPBuf = uniformU32(device, [0, d]);
    }
    _allocActivations(arch) {
        const { device } = this;
        const { d_model: d, d_ff: ff, n_layers: nl, max_len: ctx, vocab_size: vs } = arch;
        const act = (n) => emptyStorageBuf(device, n * 4, BU_COPY_SRC);
        this.xBuf = act(d);
        this.lnBuf = act(d);
        this.qBuf = act(d);
        this.kvBuf = act(d);
        this.attnBuf = act(d);
        this.ffBuf = act(ff);
        this.logBuf = act(vs);
        this.stagBuf = device.createBuffer({
            size: vs * 4, usage: BU_MAP_READ | BU_COPY_DST,
        });
        const cacheBytes = ctx * d * 4;
        for (let li = 0; li < nl; li++) {
            this.kCache.push(emptyStorageBuf(device, cacheBytes));
            this.vCache.push(emptyStorageBuf(device, cacheBytes));
        }
    }
    // ── Public API ─────────────────────────────────────────────────────────────
    /** Prefill KV cache with all prompt tokens. Returns logits for last position. */
    async prefill(tokens) {
        this._seqLen = 0;
        let logits;
        for (const tok of tokens)
            logits = await this._step(tok);
        return logits;
    }
    /** Single incremental step; returns logits[vocab_size]. */
    async step(token) {
        return this._step(token);
    }
    /** Clear the KV cache (start a new conversation). */
    reset() { this._seqLen = 0; }
    /** Release all GPU resources. */
    destroy() {
        const all = [
            this.xBuf, this.lnBuf, this.qBuf, this.kvBuf, this.attnBuf, this.ffBuf,
            this.logBuf, this.stagBuf, this.tokEmb, this.posEmb, this.headW,
            this.lnFw, this.lnFb, this.zeroBuf,
            this.embedPBuf, this.attnPBuf, this.wkvPBuf,
            this.lnPBuf, this.addPBuf, this.reluPBuf,
            this.matQKVOPBuf, this.matFF1PBuf, this.matFF2PBuf, this.headPBuf,
            ...this.ln1w, ...this.ln1b, ...this.ln2w, ...this.ln2b,
            ...this.kCache, ...this.vCache,
        ];
        for (const b of all)
            b?.destroy();
        for (const t of [...this.qT, ...this.kT, ...this.vT, ...this.oT, ...this.ff1T, ...this.ff2T]) {
            t.f32Buf?.destroy();
            t.nibBuf?.destroy();
            t.sclBuf?.destroy();
            t.int8Buf?.destroy();
            t.paramBuf?.destroy();
            t.biasBuf.destroy();
        }
        this.device.destroy();
    }
    // ── Core inference step ───────────────────────────────────────────────────
    async _step(token) {
        const { device, arch, pip } = this;
        const { d_model: d, n_heads: nh, n_layers: nl, d_ff: ff, vocab_size: vs } = arch;
        const dh = d / nh;
        const pos = this._seqLen;
        const seqLen = pos + 1;
        // Update per-token uniforms
        device.queue.writeBuffer(this.embedPBuf, 0, new Uint32Array([token, pos, d]));
        device.queue.writeBuffer(this.wkvPBuf, 0, new Uint32Array([pos, d]));
        device.queue.writeBuffer(this.attnPBuf, 0, new Uint32Array([seqLen, nh, dh, d]));
        const enc = device.createCommandEncoder();
        // 1. Token + positional embedding → xBuf
        this._passEmbed(enc);
        // 2. Transformer layers
        for (let li = 0; li < nl; li++) {
            // Attention block
            this._passLN(enc, this.xBuf, this.ln1w[li], this.ln1b[li], this.lnBuf);
            this._passMat(enc, this.qT[li], this.lnBuf, this.qBuf);
            this._passMat(enc, this.kT[li], this.lnBuf, this.kvBuf);
            this._passWKV(enc, this.kvBuf, this.kCache[li]);
            this._passMat(enc, this.vT[li], this.lnBuf, this.kvBuf);
            this._passWKV(enc, this.kvBuf, this.vCache[li]);
            this._passAttn(enc, li);
            this._passMat(enc, this.oT[li], this.attnBuf, this.lnBuf);
            this._passAdd(enc, this.xBuf, this.lnBuf);
            // FFN block
            this._passLN(enc, this.xBuf, this.ln2w[li], this.ln2b[li], this.lnBuf);
            this._passMat(enc, this.ff1T[li], this.lnBuf, this.ffBuf);
            this._passReLU(enc);
            this._passMat(enc, this.ff2T[li], this.ffBuf, this.lnBuf);
            this._passAdd(enc, this.xBuf, this.lnBuf);
        }
        // 3. Final layernorm + head projection
        this._passLN(enc, this.xBuf, this.lnFw, this.lnFb, this.lnBuf);
        this._passHead(enc);
        // 4. Copy logits to staging buffer and submit
        enc.copyBufferToBuffer(this.logBuf, 0, this.stagBuf, 0, vs * 4);
        device.queue.submit([enc.finish()]);
        // 5. Read back logits (only 258 × 4 = ~1 KB)
        await this.stagBuf.mapAsync(GPU_MAP_READ);
        const result = new Float32Array(this.stagBuf.getMappedRange().slice(0));
        this.stagBuf.unmap();
        // Debug: log logit stats for the first step of each generate call
        if (pos === 0) {
            let mn = Infinity, mx = -Infinity, nans = 0;
            for (let i = 0; i < result.length; i++) {
                if (isNaN(result[i])) {
                    nans++;
                }
                else {
                    mn = Math.min(mn, result[i]);
                    mx = Math.max(mx, result[i]);
                }
            }
            const top3 = Array.from(result).map((v, i) => [i, v])
                .sort((a, b) => b[1] - a[1]).slice(0, 3);
            console.log(`[gpu] pos=0 logits: min=${mn.toFixed(3)} max=${mx.toFixed(3)} nans=${nans} top3=`, top3);
        }
        this._seqLen = seqLen;
        return result;
    }
    // ── Compute pass helpers ──────────────────────────────────────────────────
    _bg(pipeline, buffers) {
        return this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
        });
    }
    _passEmbed(enc) {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pip.embed);
        pass.setBindGroup(0, this._bg(this.pip.embed, [this.tokEmb, this.posEmb, this.xBuf, this.embedPBuf]));
        pass.dispatchWorkgroups(Math.ceil(this.arch.d_model / 256));
        pass.end();
    }
    _passLN(enc, x, w, b, y) {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pip.ln);
        pass.setBindGroup(0, this._bg(this.pip.ln, [x, w, b, y, this.lnPBuf]));
        pass.dispatchWorkgroups(1); // single workgroup reduction for d ≤ 512
        pass.end();
    }
    _passMat(enc, wt, input, output) {
        const { d_model: d, d_ff: ff } = this.arch;
        const wgs = Math.ceil(wt.outDim / 256);
        const pass = enc.beginComputePass();
        if (wt.dtype === 'int4g') {
            const pb = (wt.outDim === ff && wt.inDim === d) ? this.matFF1PBuf
                : (wt.outDim === d && wt.inDim === ff) ? this.matFF2PBuf
                    : this.matQKVOPBuf;
            pass.setPipeline(this.pip.matI4g);
            pass.setBindGroup(0, this._bg(this.pip.matI4g, [wt.nibBuf, wt.sclBuf, input, wt.biasBuf, output, pb]));
        }
        else if (wt.dtype === 'int8') {
            pass.setPipeline(this.pip.matI8);
            pass.setBindGroup(0, this._bg(this.pip.matI8, [wt.int8Buf, input, wt.biasBuf, output, wt.paramBuf]));
        }
        else {
            const pb = (wt.outDim === ff && wt.inDim === d) ? this.matFF1PBuf
                : (wt.outDim === d && wt.inDim === ff) ? this.matFF2PBuf
                    : this.matQKVOPBuf;
            pass.setPipeline(this.pip.matF32);
            pass.setBindGroup(0, this._bg(this.pip.matF32, [wt.f32Buf, input, wt.biasBuf, output, pb]));
        }
        pass.dispatchWorkgroups(wgs);
        pass.end();
    }
    _passWKV(enc, src, dst) {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pip.writeKV);
        pass.setBindGroup(0, this._bg(this.pip.writeKV, [src, dst, this.wkvPBuf]));
        pass.dispatchWorkgroups(Math.ceil(this.arch.d_model / 256));
        pass.end();
    }
    _passAttn(enc, li) {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pip.attn);
        pass.setBindGroup(0, this._bg(this.pip.attn, [this.qBuf, this.kCache[li], this.vCache[li], this.attnBuf, this.attnPBuf]));
        pass.dispatchWorkgroups(this.arch.n_heads);
        pass.end();
    }
    _passAdd(enc, a, b) {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pip.add);
        pass.setBindGroup(0, this._bg(this.pip.add, [a, b, this.addPBuf]));
        pass.dispatchWorkgroups(Math.ceil(this.arch.d_model / 256));
        pass.end();
    }
    _passReLU(enc) {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pip.relu);
        pass.setBindGroup(0, this._bg(this.pip.relu, [this.ffBuf, this.reluPBuf]));
        pass.dispatchWorkgroups(Math.ceil(this.arch.d_ff / 256));
        pass.end();
    }
    _passHead(enc) {
        // Head projection is always fp32 (head_weight stored as float32)
        const { vocab_size: vs } = this.arch;
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pip.matF32);
        pass.setBindGroup(0, this._bg(this.pip.matF32, [this.headW, this.lnBuf, this.zeroBuf, this.logBuf, this.headPBuf]));
        pass.dispatchWorkgroups(Math.ceil(vs / 256));
        pass.end();
    }
}
//# sourceMappingURL=gpu_engine.js.map