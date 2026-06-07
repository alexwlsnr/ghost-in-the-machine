/**
 * Tier 2.5 Ghost Transformer — Float32 Orchestrator (fixed lengths)
 */

interface SectionDef { offset: number; size: number; shape: number[]; dtype: string; }
interface Arch { vocab_size: number; d_model: number; n_heads: number; n_layers: number; d_ff: number; max_len: number; }

interface WasmApi {
  memory: WebAssembly.Memory;
  matmul_f32w(w: number, b: number, inp: number, out: number, inD: number, outD: number): void;
  softmax_f32(p: number, n: number): void;
  softmax_causal_f32(p: number, s: number): void;
  layer_norm_f32(x: number, g: number, b: number, d: number, e: number): void;
  add_vec_f32(a: number, b: number, n: number): void;
  relu_f32(p: number, n: number): void;
}

const PAD = 256;
const EOS = 257;

async function fetchBuf(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
  return r.arrayBuffer();
}

export async function loadModel(urls: { wasm: string; bin: string; json: string }) {
  const [wasmBuf, binBuf, jsonBuf] = await Promise.all([
    fetchBuf(urls.wasm), fetchBuf(urls.bin), fetchBuf(urls.json),
  ]);
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
  const needPages = Math.ceil((base + maxOff + 8 * 1024 * 1024) / 65536);
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

// ─── Forward ───────────────────────────────────────────────────────

function forward(api: WasmApi, sec: Record<string, SectionDef>, arch: Arch, tokens: number[], base: number): Float32Array {
  const d = arch.d_model, nh = arch.n_heads, dh = d / nh, nl = arch.n_layers, seq = tokens.length, mem = api.memory;

  // Section pointer helper: actual address = base + manifest offset
  const S = (name: string) => base + sec[name].offset;

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
  const oOff = ba(arch.vocab_size + d);

  // 1. Embedding
  const teW = f32(S('token_embed'), 257 * d);
  const peW = f32(S('pos_embed'), 64 * d);
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
      const lnBuf = f32(lOff, d);
      for (let j = 0; j < d; j++) lnBuf[j] = emb[p * d + j];
      api.layer_norm_f32(lOff, S(`${pfx}_ln1_w`), S(`${pfx}_ln1_b`), d, 1e-5);
      const qp = qOff + p * d * 3 * 4;
      api.matmul_f32w(S(`${pfx}_q_weight`), S(`${pfx}_q_bias`), lOff, qp, d, d);
      api.matmul_f32w(S(`${pfx}_k_weight`), S(`${pfx}_k_bias`), lOff, qp + d * 4, d, d);
      api.matmul_f32w(S(`${pfx}_v_weight`), S(`${pfx}_v_bias`), lOff, qp + d * 8, d, d);
    }

    const qkv = f32(qOff, seq * d * 3);
    const attn = f32(aOff, seq * d);
    const scores = f32(sOff, seq * seq);
    attn.fill(0);

    for (let h = 0; h < nh; h++) {
      const ho = h * dh;
      for (let qi = 0; qi < seq; qi++) {
        for (let kj = 0; kj <= qi; kj++) {
          let dot = 0;
          for (let x = 0; x < dh; x++) dot += qkv[qi * d * 3 + ho + x] * qkv[kj * d * 3 + d + ho + x];
          scores[qi * seq + kj] = dot / Math.sqrt(dh);
        }
        for (let kj = qi + 1; kj < seq; kj++) scores[qi * seq + kj] = -Infinity;
      }
      api.softmax_causal_f32(sOff, seq);
      for (let qi = 0; qi < seq; qi++) {
        for (let x = 0; x < dh; x++) {
          let val = 0;
          for (let kj = 0; kj < seq; kj++) val += scores[qi * seq + kj] * qkv[kj * d * 3 + d * 2 + ho + x];
          attn[qi * d + ho + x] = val;
        }
      }
    }

    for (let p = 0; p < seq; p++) {
      api.matmul_f32w(S(`${pfx}_o_weight`), S(`${pfx}_o_bias`), aOff + p * d * 4, tOff + p * d * 4, d, d);
    }
    api.add_vec_f32(eOff, tOff, seq * d);

    // FFN
    for (let p = 0; p < seq; p++) {
      const lnBuf = f32(lOff, d);
      for (let j = 0; j < d; j++) lnBuf[j] = emb[p * d + j];
      api.layer_norm_f32(lOff, S(`${pfx}_ln2_w`), S(`${pfx}_ln2_b`), d, 1e-5);
      const up = tOff + p * arch.d_ff * 4;
      api.matmul_f32w(S(`${pfx}_ff1_weight`), S(`${pfx}_ff1_bias`), lOff, up, d, arch.d_ff);
      api.relu_f32(up, arch.d_ff);
      api.matmul_f32w(S(`${pfx}_ff2_weight`), S(`${pfx}_ff2_bias`), up, lOff, arch.d_ff, d);
      for (let j = 0; j < d; j++) emb[p * d + j] += f32(lOff, d)[j];
    }
  }

  // 3. Final LN + head
  const lp = seq - 1;
  const lnBuf = f32(lOff, d);
  for (let j = 0; j < d; j++) lnBuf[j] = emb[lp * d + j];
  api.layer_norm_f32(lOff, S('lnf_w'), S('lnf_b'), d, 1e-5);

  const zb = f32(oOff, d);
  zb.fill(0);
  const lgOff = oOff + d * 4;
  api.matmul_f32w(S('head_weight'), oOff, lOff, lgOff, d, arch.vocab_size);
  return f32(lgOff, arch.vocab_size);
}

// ─── Generation ────────────────────────────────────────────────────

export interface Step { char: string; token: number; done: boolean; }

export async function* generate(
  model: Awaited<ReturnType<typeof loadModel>>,
  prompt: string, maxNew = 160, temp = 0.8,
): AsyncGenerator<Step> {
  const { api, manifest: arch, sec, base } = model;
  const win = arch.max_len - 1;            // hard context limit of the model
  const tokens = encode(prompt.toUpperCase()).slice(0, win);

  for (let s = 0; s < maxNew; s++) {
    // Model can only attend to `win` tokens; beyond that it produces garbage,
    // so stop cleanly rather than sliding the window.
    if (tokens.length >= win) { yield { char: '', token: PAD, done: true }; return; }

    // Yield to the browser so it can paint: makes the prompt + thinking
    // indicator appear immediately, and streams each token as it arrives.
    await new Promise((r) => setTimeout(r, 0));

    const logits = forward(api, sec, arch, tokens, base);

    let maxV = -Infinity;
    for (let i = 0; i < arch.vocab_size; i++) if (logits[i] > maxV) maxV = logits[i];
    let sum = 0;
    const probs = new Float64Array(arch.vocab_size);
    for (let i = 0; i < arch.vocab_size; i++) { probs[i] = Math.exp((logits[i] - maxV) / temp); sum += probs[i]; }
    let r = Math.random() * sum, next = 0;
    for (let i = 0; i < arch.vocab_size; i++) { r -= probs[i]; if (r <= 0) { next = i; break; } }

    if (next === EOS || next === PAD) { yield { char: '', token: next, done: true }; return; }
    tokens.push(next);
    yield { char: next < 256 ? String.fromCharCode(next) : '', token: next, done: false };
  }
  yield { char: '', token: PAD, done: true };
}
