/**
 * Tier 2.5 Ghost Transformer — Float32 Orchestrator (fixed lengths)
 */

import type { GPUEngine } from './gpu_engine.js';
import { BPETokenizer } from './bpe_tokenizer.js';
import type { BPEData } from './bpe_tokenizer.js';

export interface SectionDef { offset: number; size: number; shape: number[]; dtype: string; scale?: number; scales_offset?: number; scales_size?: number; group_size?: number; }
export interface Arch { vocab_size: number; d_model: number; n_heads: number; n_layers: number; d_ff: number; max_len: number;
  // Modern architecture flags (absent = false / classic)
  arch?: string; use_rope?: boolean; use_swiglu?: boolean; use_rmsnorm?: boolean; }

interface WasmApi {
  memory: WebAssembly.Memory;
  matmul_ternary(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
  matmul_ternary_simd(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
  matmul_ternary_sparse(counts: number, pos: number, neg: number, scale: number, b: number, inp: number, out: number, outD: number): void;
  ternary_convert_to_sparse(w: number, counts: number, pos: number, neg: number, inD: number, outD: number): void;
  matmul_f32w(w: number, b: number, inp: number, out: number, inD: number, outD: number): void;
  softmax_f32(p: number, n: number): void;
  softmax_causal_f32(p: number, s: number): void;
  layer_norm_f32(x: number, g: number, b: number, d: number, e: number): void;
  rms_norm_f32(x: number, g: number, d: number, e: number): void;
  add_vec_f32(a: number, b: number, n: number): void;
  mul_vec_f32(a: number, b: number, n: number): void;
  relu_f32(p: number, n: number): void;
  silu_f32(p: number, n: number): void;
  attention_f32(qkv: number, scores: number, attn: number, seq: number, d: number, nHeads: number): void;
  matmul_8bit(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
  matmul_4bit(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
  matmul_4bit_grouped(w: number, scales: number, b: number, inp: number, out: number, inD: number, outD: number, groupSize: number): void;
  matmul_bf16(w: number, b: number, inp: number, out: number, inD: number, outD: number): void;
}

const PAD = 256;
const SEP = 1;    // ASCII SOH — query/response separator (matches Python SEP_TOKEN)
const EOS = 257;

async function fetchBuf(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
  return r.arrayBuffer();
}

// Worst-case scratch the forward pass bump-allocates (bytes) at full context.
// Mirrors the `ba(...)` allocations in forward() — keep the two in sync.
// SwiGLU needs 2×d_ff scratch per position (gate + value buffers).
function forwardScratchBytes(arch: Arch): number {
  const d = arch.d_model, ff = arch.d_ff, v = arch.vocab_size, seq = arch.max_len;
  const al = (n: number) => (n * 4 + 15) & ~15;
  const maxFF = arch.use_swiglu ? ff * 2 : ff;
  return al(seq * d) + al(seq * d * 3) + al(seq * Math.max(d, maxFF))
       + al(d) + al(seq * seq) + al(seq * d) + al(v * 2);
}

/** Detect Wasm SIMD128 support at runtime (tiny probe Wasm module). */
export async function detectSIMD(): Promise<boolean> {
  // Minimal Wasm module that uses a SIMD instruction (f32x4.splat).
  // If the browser/runtime supports SIMD128, WebAssembly.validate returns true.
  try {
    return WebAssembly.validate(new Uint8Array([
      0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,
      10,10,1,8,0,65,0,253,17,253,98,11
    ]));
  } catch { return false; }
}

export async function loadModel(urls: { wasm: string; bin: string; json: string }) {
  const [wasmBuf, binBuf, jsonBuf] = await Promise.all([
    fetchBuf(urls.wasm), fetchBuf(urls.bin), fetchBuf(urls.json),
  ]);
  return instantiateModel(wasmBuf, binBuf, jsonBuf);
}

// Construct a model from already-loaded buffers (no fetch). Split out of loadModel
// so the forward pass can be driven under Node for parity tests.
export async function instantiateModel(wasmBuf: BufferSource, binBuf: ArrayBuffer, jsonBuf: ArrayBuffer) {
  const wasm = await WebAssembly.instantiate(wasmBuf);
  const api = wasm.instance.exports as unknown as WasmApi;
  const manifest = JSON.parse(new TextDecoder().decode(jsonBuf)) as {
    architecture: Arch; sections: Record<string, SectionDef>;
    tokenizer?: BPEData;
  };
  const bpe = manifest.tokenizer ? new BPETokenizer(manifest.tokenizer) : undefined;
  const sec = manifest.sections;
  const mem = api.memory;

  // CRITICAL: Wasm module uses memory [0, __heap_base) for its own stack/data.
  // We must place model weights at __heap_base or higher, otherwise Rust
  // function stack writes will corrupt the weights.
  const heapBase = ((wasm.instance.exports as any).__heap_base?.value ?? 0) as number;
  const base = (heapBase + 15) & ~15;

  // Use the full binary size for memory allocation — this correctly handles
  // formats like int4g where per-group scales are appended after the weight
  // nibbles and would be missed if we only summed s.offset + s.size.
  const binSize = binBuf.byteLength;
  const margin = forwardScratchBytes(manifest.architecture) + 65536;
  const needPages = Math.ceil((base + binSize + margin) / 65536);
  const curPages = mem.buffer.byteLength / 65536;
  if (needPages > curPages) mem.grow(needPages - curPages);

  // Copy the entire model binary in one shot so all scale arrays land correctly.
  const mem8 = new Uint8Array(mem.buffer);
  const bin8 = new Uint8Array(binBuf);
  mem8.set(bin8, base);

  // Load-time sparse conversion for ternary models.
  // CRITICAL: sparse buffers must live AFTER the forward-pass scratch area.
  // The forward() ba() allocator starts at base+binSize and grows upward —
  // placing sparse buffers there would cause the scratch to overwrite them.
  // Layout: [model binary][forward scratch margin][sparse index lists]
  const sparseBuffers = new Map<string, { countsPtr: number; posPtr: number; negPtr: number }>();
  if (api.ternary_convert_to_sparse) {
    // Compute worst-case total sparse size so we can grow memory once
    let totalSparseBytes = 0;
    for (const s of Object.values(sec) as SectionDef[]) {
      if (s.dtype !== 'ternary') continue;
      const [outD, inD] = s.shape;
      totalSparseBytes += outD * 2 * 4 + outD * inD * 4 + 16; // counts + worst-case pos+neg + align
    }

    // Sparse area starts after model binary AND forward scratch — no overlap possible
    let sparseBase = base + binSize + margin;
    sparseBase = (sparseBase + 15) & ~15;

    const sparseEnd = sparseBase + totalSparseBytes;
    const curPages2 = mem.buffer.byteLength / 65536;
    const needPages2 = Math.ceil(sparseEnd / 65536);
    if (needPages2 > curPages2) mem.grow(needPages2 - curPages2);

    for (const [name, s] of Object.entries(sec) as [string, SectionDef][]) {
      if (s.dtype !== 'ternary') continue;
      const [outD, inD] = s.shape;
      const maxNonZero = outD * inD;
      const countsPtr = sparseBase;
      const posPtr    = countsPtr + outD * 2 * 4;
      const negPtr    = posPtr    + maxNonZero * 2;
      api.ternary_convert_to_sparse(base + s.offset, countsPtr, posPtr, negPtr, inD, outD);
      sparseBuffers.set(name, { countsPtr, posPtr, negPtr });
      sparseBase = (negPtr + maxNonZero * 2 + 15) & ~15;
    }
  }

  return { api, manifest: manifest.architecture, sec, base, sparseBuffers, bpe };
}

export function encode(text: string): number[] {
  const t: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 256) t.push(c);
  }
  return t;
}

// ─── Matmul dispatch ───────────────────────────────────────────────
// Routes weight matmul to the correct kernel based on section dtype.
// Biases are always fp32. Head weight stays fp32 (mixed-precision layout).
function makeMatmulDispatch(
  api: WasmApi,
  sec: Record<string, SectionDef>,
  base: number,
  sparseBuffers?: Map<string, { countsPtr: number; posPtr: number; negPtr: number }>,
) {
  const S = (name: string) => base + sec[name].offset;
  const fp32mw    = (api as any).matmul_f32w_simd ?? api.matmul_f32w;
  const ternaryMw = (api as any).matmul_ternary_simd ?? api.matmul_ternary;
  return (wName: string, bPtr: number, inp: number, out: number, inD: number, outD: number) => {
    const s = sec[wName];
    const wPtr = S(wName);
    if (s.dtype === 'ternary') {
      const sp = sparseBuffers?.get(wName);
      if (sp && api.matmul_ternary_sparse) {
        api.matmul_ternary_sparse(sp.countsPtr, sp.posPtr, sp.negPtr, s.scale ?? 1.0, bPtr, inp, out, outD);
      } else {
        ternaryMw(wPtr, s.scale ?? 1.0, bPtr, inp, out, inD, outD);
      }
    } else if (s.dtype === 'int8')   api.matmul_8bit(wPtr, s.scale ?? 1.0, bPtr, inp, out, inD, outD);
    else if (s.dtype === 'int4')     api.matmul_4bit(wPtr, s.scale ?? 1.0, bPtr, inp, out, inD, outD);
    else if (s.dtype === 'int4g')    api.matmul_4bit_grouped(wPtr, base + s.scales_offset!, bPtr, inp, out, inD, outD, s.group_size ?? 32);
    else if (s.dtype === 'bfloat16') api.matmul_bf16(wPtr, bPtr, inp, out, inD, outD);
    else                             fp32mw(wPtr, bPtr, inp, out, inD, outD);
  };
}

// ─── Forward ───────────────────────────────────────────────────────

// ── RoPE helper (computed in JS — no Wasm needed) ────────────────────────
function precomputeRoPE(dHead: number, maxLen: number): { cos: Float32Array; sin: Float32Array } {
  const half = dHead >> 1;
  const cos = new Float32Array(maxLen * half);
  const sin = new Float32Array(maxLen * half);
  for (let p = 0; p < maxLen; p++) {
    for (let i = 0; i < half; i++) {
      const theta = p / Math.pow(10000, (2 * i) / dHead);
      cos[p * half + i] = Math.cos(theta);
      sin[p * half + i] = Math.sin(theta);
    }
  }
  return { cos, sin };
}

function applyRoPEToVec(vec: Float32Array, pos: number, cos: Float32Array, sin: Float32Array, dHead: number): void {
  // PyTorch view_as_complex pairs consecutive elements: (vec[0],vec[1]), (vec[2],vec[3])...
  const half = dHead >> 1;
  for (let i = 0; i < half; i++) {
    const x0 = vec[i * 2], x1 = vec[i * 2 + 1];
    const c = cos[pos * half + i], s = sin[pos * half + i];
    vec[i * 2]     = x0 * c - x1 * s;
    vec[i * 2 + 1] = x0 * s + x1 * c;
  }
}

// Cache RoPE frequencies per (d_head, max_len) pair
const ropeCache = new Map<string, { cos: Float32Array; sin: Float32Array }>();
function getRoPE(dHead: number, maxLen: number) {
  const key = `${dHead}_${maxLen}`;
  if (!ropeCache.has(key)) ropeCache.set(key, precomputeRoPE(dHead, maxLen));
  return ropeCache.get(key)!;
}

type SparseMap = Map<string, { countsPtr: number; posPtr: number; negPtr: number }>;

export function forward(api: WasmApi, sec: Record<string, SectionDef>, arch: Arch, tokens: number[], base: number, sparseBuffers?: SparseMap): Float32Array {
  const d = arch.d_model, nh = arch.n_heads, dh = d / nh, nl = arch.n_layers, seq = tokens.length, mem = api.memory;
  const useRope    = !!arch.use_rope;
  const useSwiglu  = !!arch.use_swiglu;
  const useRmsnorm = !!arch.use_rmsnorm;

  // Section pointer helper: actual address = base + manifest offset
  const S = (name: string) => base + sec[name].offset;
  const mw = makeMatmulDispatch(api, sec, base, sparseBuffers);

  let off = base;
  for (const s of Object.values(sec)) {
    const end = base + s.offset + s.size + (s.scales_size ?? 0);
    if (end > off) off = end;
  }
  off = (off + 15) & ~15;

  const ba = (n: number) => { const o = off; off = (off + n * 4 + 15) & ~15; return o; };
  const f32 = (o: number, n: number) => new Float32Array(mem.buffer, o, n);

  const eOff = ba(seq * d);
  const qOff = ba(seq * d * 3);
  // For SwiGLU we need 2×d_ff scratch (gate + val); for classic ReLU just d_ff
  const ffScratch = useSwiglu ? arch.d_ff * 2 : arch.d_ff;
  const tOff = ba(seq * Math.max(d, ffScratch));
  const lOff = ba(d);
  const sOff = ba(seq * seq);
  const aOff = ba(seq * d);
  const oOff = ba(arch.vocab_size * 2);

  const rope = useRope ? getRoPE(dh, arch.max_len) : null;

  // 1. Embedding
  const teW = f32(S('token_embed'), arch.vocab_size * d);
  const emb = f32(eOff, seq * d);
  if (useRope) {
    // RoPE: no positional embedding addition
    for (let p = 0; p < seq; p++) {
      const tid = tokens[p];
      for (let j = 0; j < d; j++) emb[p * d + j] = teW[tid * d + j];
    }
  } else {
    const peW = f32(S('pos_embed'), arch.max_len * d);
    for (let p = 0; p < seq; p++) {
      const tid = tokens[p];
      for (let j = 0; j < d; j++) emb[p * d + j] = teW[tid * d + j] + peW[p * d + j];
    }
  }

  // Normalisation function — routes to RMSNorm or LayerNorm
  const applyNorm = (x: number, w: number, b: number) => {
    if (useRmsnorm) api.rms_norm_f32(x, w, d, 1e-5);
    else            api.layer_norm_f32(x, w, b, d, 1e-5);
  };

  // 2. Layers
  for (let li = 0; li < nl; li++) {
    const pfx = `enc${li}`;

    // Attention pre-norm + QKV
    for (let p = 0; p < seq; p++) {
      f32(lOff, d).set(new Float32Array(mem.buffer, eOff + p * d * 4, d));
      applyNorm(lOff, S(`${pfx}_ln1_w`), S(`${pfx}_ln1_b`));
      const qp = qOff + p * d * 3 * 4;
      mw(`${pfx}_q_weight`, S(`${pfx}_q_bias`), lOff, qp,          d, d);
      mw(`${pfx}_k_weight`, S(`${pfx}_k_bias`), lOff, qp + d * 4,  d, d);
      mw(`${pfx}_v_weight`, S(`${pfx}_v_bias`), lOff, qp + d * 8,  d, d);

      // RoPE: rotate Q and K for this position
      if (rope) {
        const qVec = f32(qp, d);
        const kVec = f32(qp + d * 4, d);
        for (let h = 0; h < nh; h++) {
          applyRoPEToVec(qVec.subarray(h * dh, h * dh + dh) as Float32Array, p, rope.cos, rope.sin, dh);
          applyRoPEToVec(kVec.subarray(h * dh, h * dh + dh) as Float32Array, p, rope.cos, rope.sin, dh);
        }
      }
    }

    api.attention_f32(qOff, sOff, aOff, seq, d, nh);
    for (let p = 0; p < seq; p++) {
      mw(`${pfx}_o_weight`, S(`${pfx}_o_bias`), aOff + p * d * 4, tOff + p * d * 4, d, d);
    }
    api.add_vec_f32(eOff, tOff, seq * d);

    // FFN
    for (let p = 0; p < seq; p++) {
      f32(lOff, d).set(new Float32Array(mem.buffer, eOff + p * d * 4, d));
      applyNorm(lOff, S(`${pfx}_ln2_w`), S(`${pfx}_ln2_b`));

      if (useSwiglu) {
        // SwiGLU: gate = matmul(x, w1), val = matmul(x, w2)
        //         output = silu(gate) * val → matmul(output, w3)
        const gateOff = tOff + p * ffScratch * 4;
        const valOff  = gateOff + arch.d_ff * 4;
        mw(`${pfx}_ff_gate_weight`, S(`${pfx}_ff1_bias`), lOff, gateOff, d, arch.d_ff);
        mw(`${pfx}_ff_val_weight`,  S(`${pfx}_ff1_bias`), lOff, valOff,  d, arch.d_ff);
        api.silu_f32(gateOff, arch.d_ff);
        api.mul_vec_f32(gateOff, valOff, arch.d_ff);
        mw(`${pfx}_ff2_weight`, S(`${pfx}_ff2_bias`), gateOff, lOff, arch.d_ff, d);
      } else {
        const up = tOff + p * arch.d_ff * 4;
        mw(`${pfx}_ff1_weight`, S(`${pfx}_ff1_bias`), lOff, up, d, arch.d_ff);
        api.relu_f32(up, arch.d_ff);
        mw(`${pfx}_ff2_weight`, S(`${pfx}_ff2_bias`), up, lOff, arch.d_ff, d);
      }
      api.add_vec_f32(eOff + p * d * 4, lOff, d);
    }
  }

  // 3. Final norm + head
  const lp = seq - 1;
  f32(lOff, d).set(new Float32Array(mem.buffer, eOff + lp * d * 4, d));
  applyNorm(lOff, S('lnf_w'), S('lnf_b'));

  const zb = f32(oOff, arch.vocab_size);
  zb.fill(0);
  const lgOff = oOff + arch.vocab_size * 4;
  api.matmul_f32w(S('head_weight'), oOff, lOff, lgOff, d, arch.vocab_size);
  return f32(lgOff, arch.vocab_size);
}

// ─── KV Cache ──────────────────────────────────────────────────────

export type LoadedModel = Awaited<ReturnType<typeof loadModel>>;

export interface KVCache {
  /** Number of positions already written into the cache. */
  length: number;
  /** Per-layer K buffer: [max_len * d_model] floats. Index: layer → position * d + dim */
  k: Float32Array[];
  /** Per-layer V buffer: [max_len * d_model] floats. */
  v: Float32Array[];
}

export function createCache(model: LoadedModel): KVCache {
  const { manifest: arch } = model;
  const size = arch.max_len * arch.d_model;
  return {
    length: 0,
    k: Array.from({ length: arch.n_layers }, () => new Float32Array(size)),
    v: Array.from({ length: arch.n_layers }, () => new Float32Array(size)),
  };
}

// ─── Incremental Forward (KV Cache path) ───────────────────────────

/**
 * Single-token incremental forward using a KV cache.
 * Computes logits for position `pos` (0-indexed), updating the cache.
 * The cache must already contain filled entries for positions 0..pos-1.
 *
 * On entry: cache.length === pos
 * On exit:  cache.length === pos + 1
 */
function forwardIncremental(
  api: WasmApi,
  sec: Record<string, SectionDef>,
  arch: Arch,
  token: number,
  pos: number,
  base: number,
  cache: KVCache,
  sparseBuffers?: SparseMap,
): Float32Array {
  const d = arch.d_model, nh = arch.n_heads, dh = d / nh, nl = arch.n_layers;
  const mem = api.memory;
  const useRope    = !!arch.use_rope;
  const useSwiglu  = !!arch.use_swiglu;
  const useRmsnorm = !!arch.use_rmsnorm;
  const S = (name: string) => base + sec[name].offset;
  const mw = makeMatmulDispatch(api, sec, base, sparseBuffers);
  const rope = useRope ? getRoPE(dh, arch.max_len) : null;

  let off = base;
  for (const s of Object.values(sec)) {
    const end = base + s.offset + s.size + (s.scales_size ?? 0);
    if (end > off) off = end;
  }
  off = (off + 15) & ~15;

  const ba = (n: number) => { const o = off; off = (off + n * 4 + 15) & ~15; return o; };
  const f32 = (o: number, n: number) => new Float32Array(mem.buffer, o, n);

  const xOff    = ba(d);
  const qOff    = ba(d);
  const kvOff   = ba(d);
  const aOff    = ba(d);
  const seqLen  = pos + 1;
  const scOff   = ba(seqLen);
  const lOff    = ba(d);
  // Pre-allocate FFN scratch outside the loop to avoid unbounded growth.
  const ffOff   = ba(arch.d_ff);                    // classic ReLU or SwiGLU gate
  const ffValOff = useSwiglu ? ba(arch.d_ff) : 0;  // SwiGLU value branch
  const oOff    = ba(arch.vocab_size * 2);

  const applyNorm = (x: number, w: number, b: number) => {
    if (useRmsnorm) api.rms_norm_f32(x, w, d, 1e-5);
    else            api.layer_norm_f32(x, w, b, d, 1e-5);
  };

  // 1. Embedding
  const teW = f32(S('token_embed'), arch.vocab_size * d);
  const xVec = f32(xOff, d);
  if (useRope) {
    for (let j = 0; j < d; j++) xVec[j] = teW[token * d + j];
  } else {
    const peW = f32(S('pos_embed'), arch.max_len * d);
    for (let j = 0; j < d; j++) xVec[j] = teW[token * d + j] + peW[pos * d + j];
  }

  // 2. Layers
  for (let li = 0; li < nl; li++) {
    const pfx = `enc${li}`;
    const cacheK = cache.k[li];
    const cacheV = cache.v[li];

    // Pre-norm
    const lnBuf = f32(lOff, d);
    lnBuf.set(f32(xOff, d));
    applyNorm(lOff, S(`${pfx}_ln1_w`), S(`${pfx}_ln1_b`));

    // QKV
    mw(`${pfx}_q_weight`, S(`${pfx}_q_bias`), lOff, qOff,  d, d);
    mw(`${pfx}_k_weight`, S(`${pfx}_k_bias`), lOff, kvOff, d, d);

    // RoPE: rotate Q and K for current position
    if (rope) {
      const qv = f32(qOff, d), kv = f32(kvOff, d);
      for (let h = 0; h < nh; h++) {
        applyRoPEToVec(qv.subarray(h * dh, h * dh + dh) as Float32Array, pos, rope.cos, rope.sin, dh);
        applyRoPEToVec(kv.subarray(h * dh, h * dh + dh) as Float32Array, pos, rope.cos, rope.sin, dh);
      }
    }

    cacheK.set(f32(kvOff, d), pos * d);
    mw(`${pfx}_v_weight`, S(`${pfx}_v_bias`), lOff, kvOff, d, d);
    cacheV.set(f32(kvOff, d), pos * d);

    // Attention
    const attn = f32(aOff, d);
    attn.fill(0);
    const scores = f32(scOff, seqLen);
    const q = f32(qOff, d);
    for (let h = 0; h < nh; h++) {
      const ho = h * dh;
      for (let kj = 0; kj < seqLen; kj++) {
        let dot = 0;
        for (let xi = 0; xi < dh; xi++) dot += q[ho + xi] * cacheK[kj * d + ho + xi];
        scores[kj] = dot / Math.sqrt(dh);
      }
      api.softmax_f32(scOff, seqLen);
      for (let xi = 0; xi < dh; xi++) {
        let val = 0;
        for (let kj = 0; kj < seqLen; kj++) val += scores[kj] * cacheV[kj * d + ho + xi];
        attn[ho + xi] = val;
      }
    }

    mw(`${pfx}_o_weight`, S(`${pfx}_o_bias`), aOff, lOff, d, d);
    const lVec = f32(lOff, d);
    for (let j = 0; j < d; j++) xVec[j] += lVec[j];

    // FFN
    lnBuf.set(f32(xOff, d));
    applyNorm(lOff, S(`${pfx}_ln2_w`), S(`${pfx}_ln2_b`));

    if (useSwiglu) {
      mw(`${pfx}_ff_gate_weight`, S(`${pfx}_ff1_bias`), lOff, ffOff,    d, arch.d_ff);
      mw(`${pfx}_ff_val_weight`,  S(`${pfx}_ff1_bias`), lOff, ffValOff, d, arch.d_ff);
      api.silu_f32(ffOff, arch.d_ff);
      api.mul_vec_f32(ffOff, ffValOff, arch.d_ff);
      mw(`${pfx}_ff2_weight`, S(`${pfx}_ff2_bias`), ffOff, lOff, arch.d_ff, d);
    } else {
      mw(`${pfx}_ff1_weight`, S(`${pfx}_ff1_bias`), lOff, ffOff, d, arch.d_ff);
      api.relu_f32(ffOff, arch.d_ff);
      mw(`${pfx}_ff2_weight`, S(`${pfx}_ff2_bias`), ffOff, lOff, arch.d_ff, d);
    }
    for (let j = 0; j < d; j++) xVec[j] += lVec[j];
  }

  // 3. Final norm + head
  const lnFinal = f32(lOff, d);
  lnFinal.set(f32(xOff, d));
  applyNorm(lOff, S('lnf_w'), S('lnf_b'));

  const zb = f32(oOff, arch.vocab_size);
  zb.fill(0);
  const lgOff = oOff + arch.vocab_size * 4;
  api.matmul_f32w(S('head_weight'), oOff, lOff, lgOff, d, arch.vocab_size);
  cache.length = pos + 1;
  return f32(lgOff, arch.vocab_size);
}

/**
 * Prefill the KV cache by running forwardIncremental() over each prompt token.
 * Returns the logits for the last prompt position (seeds the first sampled token).
 * cache.length will equal tokens.length after this returns.
 */
function prefill(
  api: WasmApi,
  sec: Record<string, SectionDef>,
  arch: Arch,
  tokens: number[],
  base: number,
  cache: KVCache,
  sparseBuffers?: SparseMap,
): Float32Array {
  cache.length = 0;
  let logits!: Float32Array;
  for (let p = 0; p < tokens.length; p++) {
    logits = forwardIncremental(api, sec, arch, tokens[p], p, base, cache, sparseBuffers);
  }
  return logits;
}

// ─── Generation ────────────────────────────────────────────────────

/** One turn of conversation history: the human query and the model's response. */
export interface Turn { q: string; r: string; }

/**
 * Build the token sequence for a new query, prepending conversation history.
 *
 * Layout with 1 history turn:
 *   [q_prev][SEP][r_prev][SEP][query][SEP]
 *
 * maxHistory caps how many turns are injected. Default 1 — Spec512 v1 was
 * trained on mostly 2-turn sequences so >1 history turn degrades quality.
 * Increase when a model is retrained with richer multi-turn data.
 *
 * Turns are added newest-first; oldest turns are silently dropped when the
 * context budget (maxLen) would be exceeded.
 */
export function buildContextTokens(
  history: Turn[],
  query: string,
  maxLen: number,
  maxHistory: number = 1,
  bpe?: BPETokenizer,
): number[] {
  const _enc = bpe ? (s: string) => bpe.encode(s) : encode;
  const _SEP  = bpe ? bpe.SEP : SEP;
  const queryTokens = [..._enc(query.toUpperCase()), _SEP];
  let tokens = queryTokens;

  const recent = history.slice(-maxHistory);
  for (const turn of [...recent].reverse()) {
    const chunk = [
      ..._enc(turn.q.toUpperCase()), _SEP,
      ..._enc(turn.r.toUpperCase()), _SEP,
    ];
    if (tokens.length + chunk.length >= maxLen) break;
    tokens = [...chunk, ...tokens];
  }
  return tokens;
}

export interface Step { char: string; token: number; done: boolean; }

/**
 * Sample a token index from logits with temperature + optional top-k / top-p.
 * topK=0 / topP=1.0 → pure temperature (default, backward-compatible).
 * Recommended for Shade/Specter: topK=40, topP=0.9.
 *
 * repPenalty > 1.0 penalises tokens that appeared in recentTokens by dividing
 * their logit by the penalty before softmax. 1.3 is subtle, 1.5 is aggressive.
 */
export function sampleFromLogits(
  logits: Float32Array, temp: number, topK: number, topP: number,
  rand: () => number,
  repPenalty = 1.0, recentTokens: number[] = [],
): number {
  const n = logits.length;

  // Repetition penalty: divide logits of recently-seen tokens
  if (repPenalty > 1.0 && recentTokens.length > 0) {
    const seen = new Set(recentTokens);
    for (const tok of seen) {
      if (tok < n) logits[tok] /= repPenalty;
    }
  }

  const pairs: [number, number][] = Array.from({ length: n }, (_, i) => [i, logits[i]]);
  pairs.sort((a, b) => b[1] - a[1]);
  const k = topK > 0 ? Math.min(topK, n) : n;
  const candidates = pairs.slice(0, k);
  const maxL = candidates[0][1];
  let sum = 0;
  const weighted: [number, number][] = candidates.map(([idx, l]) => {
    const p = Math.exp((l - maxL) / temp); sum += p; return [idx, p];
  });
  if (topP < 1.0) {
    let cumul = 0, cutoff = weighted.length;
    for (let i = 0; i < weighted.length; i++) {
      cumul += weighted[i][1] / sum;
      if (cumul >= topP) { cutoff = i + 1; break; }
    }
    weighted.splice(cutoff);
    sum = weighted.reduce((s, [, p]) => s + p, 0);
  }
  let r = rand() * sum;
  for (const [idx, p] of weighted) { r -= p; if (r <= 0) return idx; }
  return weighted[weighted.length - 1][0];
}

export async function* generate(
  model: Awaited<ReturnType<typeof loadModel>>,
  prompt: string, maxNew = 160, temp = 0.8, rand: () => number = Math.random,
  cache?: KVCache, topK = 0, topP = 1.0,
  history?: Turn[],
  gpuEngine?: GPUEngine,
): AsyncGenerator<Step> {
  const { api, manifest: arch, sec, base } = model;
  const bpeT = (model as any).bpe as BPETokenizer | undefined;

  // Token-type helpers — byte-level defaults when no BPE tokenizer present
  const _SEP = bpeT ? bpeT.SEP : SEP;
  const _EOS = bpeT ? bpeT.EOS : EOS;
  const _PAD = bpeT ? bpeT.PAD : PAD;
  const _enc = bpeT ? (s: string) => bpeT.encode(s) : encode;
  const _dec = bpeT
    ? (id: number) => bpeT.decodeToken(id)
    : (id: number) => (id < 256 && id !== SEP) ? String.fromCharCode(id) : '';

  const win = arch.max_len - 1;
  const tokens = (history && history.length > 0)
    ? buildContextTokens(history, prompt, win, 1, bpeT)
    : [..._enc(prompt.toUpperCase()).slice(0, win - 1), _SEP];

  const repPenalty = arch.max_len >= 512 ? 1.35 : 1.15;
  const REP_WINDOW = 32;
  const generated: number[] = [];

  // -- GPU path -------------------------------------------------------
  if (gpuEngine !== undefined) {
    gpuEngine.reset();
    let logits = await gpuEngine.prefill(tokens);
    for (let s = 0; s < maxNew; s++) {
      if (gpuEngine.seqLen >= win) { yield { char: '', token: _PAD, done: true }; return; }
      await new Promise((r) => setTimeout(r, 0));
      const next = sampleFromLogits(logits, temp, topK, topP, rand,
                                     repPenalty, generated.slice(-REP_WINDOW));
      if (next === _EOS || next === _PAD) { yield { char: '', token: next, done: true }; return; }
      generated.push(next);
      yield { char: _dec(next), token: next, done: false };
      logits = await gpuEngine.step(next);
    }
    yield { char: '', token: _PAD, done: true };
    return;
  }

  // -- Cached path ----------------------------------------------------
  if (cache !== undefined) {
    cache.length = 0;
    const sp = (model as any).sparseBuffers as SparseMap | undefined;
    let logits = prefill(api, sec, arch, tokens, base, cache, sp);
    for (let s = 0; s < maxNew; s++) {
      if (cache.length >= win) { yield { char: '', token: _PAD, done: true }; return; }
      await new Promise((r) => setTimeout(r, 0));
      const next = sampleFromLogits(logits, temp, topK, topP, rand,
                                     repPenalty, generated.slice(-REP_WINDOW));
      if (next === _EOS || next === _PAD) { yield { char: '', token: next, done: true }; return; }
      generated.push(next);
      yield { char: _dec(next), token: next, done: false };
      logits = forwardIncremental(api, sec, arch, next, cache.length, base, cache, sp);
    }
    yield { char: '', token: _PAD, done: true };
    return;
  }

  // -- Full-recompute path (no cache — backward compatible) -----------
  for (let s = 0; s < maxNew; s++) {
    if (tokens.length >= win) { yield { char: '', token: _PAD, done: true }; return; }
    await new Promise((r) => setTimeout(r, 0));
    const logits = forward(api, sec, arch, tokens, base);
    const next = sampleFromLogits(logits, temp, topK, topP, rand,
                                   repPenalty, generated.slice(-REP_WINDOW));
    if (next === _EOS || next === _PAD) { yield { char: '', token: next, done: true }; return; }
    generated.push(next);
    tokens.push(next);
    yield { char: _dec(next), token: next, done: false };
  }
  yield { char: '', token: _PAD, done: true };
}
