/**
 * Tier 2.5 Ghost Transformer — Float32 Orchestrator (fixed lengths)
 */

interface SectionDef { offset: number; size: number; shape: number[]; dtype: string; scale?: number; }
interface Arch { vocab_size: number; d_model: number; n_heads: number; n_layers: number; d_ff: number; max_len: number; }

interface WasmApi {
  memory: WebAssembly.Memory;
  matmul_f32w(w: number, b: number, inp: number, out: number, inD: number, outD: number): void;
  softmax_f32(p: number, n: number): void;
  softmax_causal_f32(p: number, s: number): void;
  layer_norm_f32(x: number, g: number, b: number, d: number, e: number): void;
  add_vec_f32(a: number, b: number, n: number): void;
  relu_f32(p: number, n: number): void;
  attention_f32(qkv: number, scores: number, attn: number, seq: number, d: number, nHeads: number): void;
  matmul_8bit(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
  matmul_4bit(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
  matmul_bf16(w: number, b: number, inp: number, out: number, inD: number, outD: number): void;
}

const PAD = 256;
const EOS = 257;

async function fetchBuf(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
  return r.arrayBuffer();
}

// Worst-case scratch the forward pass bump-allocates (bytes) at full context.
// Mirrors the `ba(...)` allocations in forward() — keep the two in sync.
function forwardScratchBytes(arch: Arch): number {
  const d = arch.d_model, ff = arch.d_ff, v = arch.vocab_size, seq = arch.max_len;
  const al = (n: number) => (n * 4 + 15) & ~15;
  return al(seq * d) + al(seq * d * 3) + al(seq * Math.max(d, ff))
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
  };
  const sec = manifest.sections;
  const mem = api.memory;

  // CRITICAL: Wasm module uses memory [0, __heap_base) for its own stack/data.
  // We must place model weights at __heap_base or higher, otherwise Rust
  // function stack writes will corrupt the weights.
  const heapBase = ((wasm.instance.exports as any).__heap_base?.value ?? 0) as number;
  const base = (heapBase + 15) & ~15;

  let maxOff = 0;
  for (const s of Object.values(sec)) maxOff = Math.max(maxOff, s.offset + s.size);
  // Headroom = the forward pass's scratch, sized from the arch (not a fixed 8 MB),
  // so larger models (more layers, longer context) get enough memory. +1 page slack.
  const margin = forwardScratchBytes(manifest.architecture) + 65536;
  const needPages = Math.ceil((base + maxOff + margin) / 65536);
  const curPages = mem.buffer.byteLength / 65536;
  if (needPages > curPages) mem.grow(needPages - curPages);

  const mem8 = new Uint8Array(mem.buffer);
  const bin8 = new Uint8Array(binBuf);
  for (const s of Object.values(sec)) {
    mem8.set(bin8.subarray(s.offset, s.offset + s.size), base + s.offset);
  }
  return { api, manifest: manifest.architecture, sec, base };
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
function makeMatmulDispatch(api: WasmApi, sec: Record<string, SectionDef>, base: number) {
  const S = (name: string) => base + sec[name].offset;
  // Use SIMD matmul for fp32 if the SIMD kernel was loaded (it exports matmul_f32w_simd).
  // Falls back transparently to the scalar matmul_f32w on non-SIMD builds.
  const fp32mw = (api as any).matmul_f32w_simd ?? api.matmul_f32w;
  return (wName: string, bPtr: number, inp: number, out: number, inD: number, outD: number) => {
    const s = sec[wName];
    const wPtr = S(wName);
    if (s.dtype === 'int8')          api.matmul_8bit(wPtr, s.scale ?? 1.0, bPtr, inp, out, inD, outD);
    else if (s.dtype === 'int4')     api.matmul_4bit(wPtr, s.scale ?? 1.0, bPtr, inp, out, inD, outD);
    else if (s.dtype === 'bfloat16') api.matmul_bf16(wPtr, bPtr, inp, out, inD, outD);
    else                             fp32mw(wPtr, bPtr, inp, out, inD, outD);
  };
}

// ─── Forward ───────────────────────────────────────────────────────

export function forward(api: WasmApi, sec: Record<string, SectionDef>, arch: Arch, tokens: number[], base: number): Float32Array {
  const d = arch.d_model, nh = arch.n_heads, dh = d / nh, nl = arch.n_layers, seq = tokens.length, mem = api.memory;

  // Section pointer helper: actual address = base + manifest offset
  const S = (name: string) => base + sec[name].offset;
  const mw = makeMatmulDispatch(api, sec, base);

  let off = base;
  for (const s of Object.values(sec)) off = Math.max(off, base + s.offset + s.size);
  off = (off + 15) & ~15;

  const ba = (n: number) => { const o = off; off = (off + n * 4 + 15) & ~15; return o; };
  const f32 = (o: number, n: number) => new Float32Array(mem.buffer, o, n);

  const eOff = ba(seq * d);
  const qOff = ba(seq * d * 3);
  const tOff = ba(seq * Math.max(d, arch.d_ff));
  const lOff = ba(d);
  const sOff = ba(seq * seq);
  const aOff = ba(seq * d);
  const oOff = ba(arch.vocab_size * 2);   // zero-bias buffer + logits, vocab_size each

  // 1. Embedding
  const teW = f32(S('token_embed'), arch.vocab_size * d);
  const peW = f32(S('pos_embed'), arch.max_len * d);
  const emb = f32(eOff, seq * d);
  for (let p = 0; p < seq; p++) {
    const tid = tokens[p];
    for (let j = 0; j < d; j++) emb[p * d + j] = teW[tid * d + j] + peW[p * d + j];
  }

  // 2. Layers
  for (let li = 0; li < nl; li++) {
    const pfx = `enc${li}`;

    // Attention: LN + QKV
    for (let p = 0; p < seq; p++) {
      // Copy emb[p] into lOff for layer norm — TypedArray.set is a native memcpy
      f32(lOff, d).set(new Float32Array(mem.buffer, eOff + p * d * 4, d));
      api.layer_norm_f32(lOff, S(`${pfx}_ln1_w`), S(`${pfx}_ln1_b`), d, 1e-5);
      const qp = qOff + p * d * 3 * 4;
      mw(`${pfx}_q_weight`, S(`${pfx}_q_bias`), lOff, qp, d, d);
      mw(`${pfx}_k_weight`, S(`${pfx}_k_bias`), lOff, qp + d * 4, d, d);
      mw(`${pfx}_v_weight`, S(`${pfx}_v_bias`), lOff, qp + d * 8, d, d);
    }

    api.attention_f32(qOff, sOff, aOff, seq, d, nh);

    for (let p = 0; p < seq; p++) {
      mw(`${pfx}_o_weight`, S(`${pfx}_o_bias`), aOff + p * d * 4, tOff + p * d * 4, d, d);
    }
    api.add_vec_f32(eOff, tOff, seq * d);

    // FFN
    for (let p = 0; p < seq; p++) {
      f32(lOff, d).set(new Float32Array(mem.buffer, eOff + p * d * 4, d));
      api.layer_norm_f32(lOff, S(`${pfx}_ln2_w`), S(`${pfx}_ln2_b`), d, 1e-5);
      const up = tOff + p * arch.d_ff * 4;
      mw(`${pfx}_ff1_weight`, S(`${pfx}_ff1_bias`), lOff, up, d, arch.d_ff);
      api.relu_f32(up, arch.d_ff);
      mw(`${pfx}_ff2_weight`, S(`${pfx}_ff2_bias`), up, lOff, arch.d_ff, d);
      // Residual add — use the existing Wasm export instead of a JS loop
      api.add_vec_f32(eOff + p * d * 4, lOff, d);
    }
  }

  // 3. Final LN + head
  const lp = seq - 1;
  f32(lOff, d).set(new Float32Array(mem.buffer, eOff + lp * d * 4, d));
  api.layer_norm_f32(lOff, S('lnf_w'), S('lnf_b'), d, 1e-5);

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
): Float32Array {
  const d = arch.d_model, nh = arch.n_heads, dh = d / nh, nl = arch.n_layers;
  const mem = api.memory;
  const S = (name: string) => base + sec[name].offset;
  const mw = makeMatmulDispatch(api, sec, base);

  // Scratch positioned after all weight sections, same convention as forward().
  let off = base;
  for (const s of Object.values(sec)) off = Math.max(off, base + s.offset + s.size);
  off = (off + 15) & ~15;

  const ba = (n: number) => { const o = off; off = (off + n * 4 + 15) & ~15; return o; };
  const f32 = (o: number, n: number) => new Float32Array(mem.buffer, o, n);

  // Fixed scratch buffers allocated before the layer loop.
  const xOff  = ba(d);                    // current hidden state for the single new position
  const qOff  = ba(d);                    // Q vector for the new position
  const kvOff = ba(d);                    // temp K or V vector, written to cache then reused
  const aOff  = ba(d);                    // attention output vector
  const seqLen = pos + 1;
  const scOff  = ba(seqLen);              // attention scores: Q[pos] attends to 0..pos
  const lOff  = ba(d);                    // temp for layernorm / ffn output
  const oOff  = ba(arch.vocab_size * 2);  // zero-bias buffer + logits

  // 1. Embedding: token + positional
  const teW = f32(S('token_embed'), arch.vocab_size * d);
  const peW = f32(S('pos_embed'), arch.max_len * d);
  const xVec = f32(xOff, d);
  for (let j = 0; j < d; j++) xVec[j] = teW[token * d + j] + peW[pos * d + j];

  // 2. Layers
  for (let li = 0; li < nl; li++) {
    const pfx = `enc${li}`;
    const cacheK = cache.k[li];   // Float32Array [max_len * d]
    const cacheV = cache.v[li];

    // ── Attention sub-layer ──────────────────────────────────────────

    // 2a. LayerNorm of x → lOff
    const lnBuf = f32(lOff, d);
    lnBuf.set(f32(xOff, d));
    api.layer_norm_f32(lOff, S(`${pfx}_ln1_w`), S(`${pfx}_ln1_b`), d, 1e-5);

    // 2b. Q/K/V for position pos
    mw(`${pfx}_q_weight`, S(`${pfx}_q_bias`), lOff, qOff, d, d);
    mw(`${pfx}_k_weight`, S(`${pfx}_k_bias`), lOff, kvOff, d, d);
    cacheK.set(f32(kvOff, d), pos * d);   // Store K[pos] into cache

    mw(`${pfx}_v_weight`, S(`${pfx}_v_bias`), lOff, kvOff, d, d);
    cacheV.set(f32(kvOff, d), pos * d);   // Store V[pos] into cache

    // 2c. Multi-head attention: Q[pos] attends to K[0..pos], V[0..pos]
    const attn = f32(aOff, d);
    attn.fill(0);
    const scores = f32(scOff, seqLen);
    const q = f32(qOff, d);

    for (let h = 0; h < nh; h++) {
      const ho = h * dh;

      // Dot products: Q[pos, h] · K[kj, h] for kj = 0..pos
      for (let kj = 0; kj < seqLen; kj++) {
        let dot = 0;
        for (let xi = 0; xi < dh; xi++) dot += q[ho + xi] * cacheK[kj * d + ho + xi];
        scores[kj] = dot / Math.sqrt(dh);
      }
      // Softmax over seqLen positions (all kj <= pos, no causal masking needed)
      api.softmax_f32(scOff, seqLen);

      // Weighted sum over V
      for (let xi = 0; xi < dh; xi++) {
        let val = 0;
        for (let kj = 0; kj < seqLen; kj++) val += scores[kj] * cacheV[kj * d + ho + xi];
        attn[ho + xi] = val;
      }
    }

    // 2d. Output projection + residual
    mw(`${pfx}_o_weight`, S(`${pfx}_o_bias`), aOff, lOff, d, d);
    const lVec = f32(lOff, d);
    for (let j = 0; j < d; j++) xVec[j] += lVec[j];

    // ── FFN sub-layer ────────────────────────────────────────────────

    // 2e. LayerNorm of x → lOff
    lnBuf.set(f32(xOff, d));
    api.layer_norm_f32(lOff, S(`${pfx}_ln2_w`), S(`${pfx}_ln2_b`), d, 1e-5);

    // 2f. ff1 (d -> d_ff), ReLU, ff2 (d_ff -> lOff as output).
    // ffOff is bump-allocated inside the layer loop; monotone bump means no aliasing.
    const ffOff = ba(arch.d_ff);
    mw(`${pfx}_ff1_weight`, S(`${pfx}_ff1_bias`), lOff, ffOff, d, arch.d_ff);
    api.relu_f32(ffOff, arch.d_ff);
    mw(`${pfx}_ff2_weight`, S(`${pfx}_ff2_bias`), ffOff, lOff, arch.d_ff, d);

    // 2g. Residual add
    for (let j = 0; j < d; j++) xVec[j] += lVec[j];
  }

  // 3. Final LN + head
  const lnFinal = f32(lOff, d);
  lnFinal.set(f32(xOff, d));
  api.layer_norm_f32(lOff, S('lnf_w'), S('lnf_b'), d, 1e-5);

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
): Float32Array {
  cache.length = 0;
  let logits!: Float32Array;
  for (let p = 0; p < tokens.length; p++) {
    logits = forwardIncremental(api, sec, arch, tokens[p], p, base, cache);
  }
  return logits;
}

// ─── Generation ────────────────────────────────────────────────────

export interface Step { char: string; token: number; done: boolean; }

/**
 * Sample a token index from logits with temperature + optional top-k / top-p.
 * topK=0 / topP=1.0 → pure temperature (default, backward-compatible).
 * Recommended for Shade/Specter: topK=40, topP=0.9.
 */
export function sampleFromLogits(
  logits: Float32Array, temp: number, topK: number, topP: number,
  rand: () => number,
): number {
  const n = logits.length;
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
): AsyncGenerator<Step> {
  const { api, manifest: arch, sec, base } = model;
  const win = arch.max_len - 1;
  const tokens = encode(prompt.toUpperCase()).slice(0, win);

  // -- Cached path ----------------------------------------------------
  if (cache !== undefined) {
    cache.length = 0;
    let logits = prefill(api, sec, arch, tokens, base, cache);
    for (let s = 0; s < maxNew; s++) {
      if (cache.length >= win) { yield { char: '', token: PAD, done: true }; return; }
      await new Promise((r) => setTimeout(r, 0));
      const next = sampleFromLogits(logits, temp, topK, topP, rand);
      if (next === EOS || next === PAD) { yield { char: '', token: next, done: true }; return; }
      yield { char: next < 256 ? String.fromCharCode(next) : '', token: next, done: false };
      logits = forwardIncremental(api, sec, arch, next, cache.length, base, cache);
    }
    yield { char: '', token: PAD, done: true };
    return;
  }

  // -- Full-recompute path (no cache — backward compatible) -----------
  for (let s = 0; s < maxNew; s++) {
    if (tokens.length >= win) { yield { char: '', token: PAD, done: true }; return; }
    await new Promise((r) => setTimeout(r, 0));
    const logits = forward(api, sec, arch, tokens, base);
    const next = sampleFromLogits(logits, temp, topK, topP, rand);
    if (next === EOS || next === PAD) { yield { char: '', token: next, done: true }; return; }
    tokens.push(next);
    yield { char: next < 256 ? String.fromCharCode(next) : '', token: next, done: false };
  }
  yield { char: '', token: PAD, done: true };
}
