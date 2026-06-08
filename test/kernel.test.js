/**
 * Wasm kernel characterization tests.
 *
 * Each op is checked against an independent JS (f64) reference implementation.
 * Run: node --test test/kernel.test.js
 */

import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const BUILT = join(ROOT, 'wasm/target/wasm32-unknown-unknown/release/tier2_kernel.wasm');
const DIST = join(ROOT, 'dist/tier2_kernel.wasm');
const WASM_PATH = existsSync(BUILT) ? BUILT : DIST;

let api;
let heap;

before(async () => {
  const buf = readFileSync(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(buf);
  api = instance.exports;
  const base = (api.__heap_base?.value ?? 0x10000) >>> 0;
  heap = (base + 15) & ~15;
});

function reset() { return heap; }
function putF32(off, arr) {
  const view = new Float32Array(api.memory.buffer, off, arr.length);
  view.set(arr);
  return { ptr: off, next: (off + arr.length * 4 + 15) & ~15 };
}
function getF32(off, n) {
  return Array.from(new Float32Array(api.memory.buffer, off, n));
}

function assertClose(actual, expected, msg, atol = 1e-4, rtol = 1e-4) {
  assert.equal(actual.length, expected.length, `${msg}: length`);
  for (let i = 0; i < actual.length; i++) {
    const tol = atol + rtol * Math.abs(expected[i]);
    assert.ok(
      Math.abs(actual[i] - expected[i]) <= tol,
      `${msg}: index ${i}: got ${actual[i]}, expected ${expected[i]} (tol ${tol})`,
    );
  }
}

function seqVals(n, seed = 1) {
  const out = new Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

function refMatmul(w, b, inp, inDim, outDim) {
  const out = new Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let sum = b[o];
    for (let i = 0; i < inDim; i++) sum += w[o * inDim + i] * inp[i];
    out[o] = sum;
  }
  return out;
}
function refSoftmax(x) {
  const m = Math.max(...x);
  const e = x.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return s > 0 ? e.map((v) => v / s) : x.map(() => 1 / x.length);
}
function refLayerNorm(x, g, b, eps) {
  const d = x.length;
  const mean = x.reduce((a, v) => a + v, 0) / d;
  const varr = x.reduce((a, v) => a + (v - mean) ** 2, 0) / d;
  const inv = 1 / Math.sqrt(varr + eps);
  return x.map((v, i) => (v - mean) * inv * g[i] + b[i]);
}
function refCausal(data, s) {
  const out = new Array(s * s).fill(0);
  for (let t = 0; t < s; t++) {
    const row = [];
    for (let j = 0; j <= t; j++) row.push(data[t * s + j]);
    const sm = refSoftmax(row);
    for (let j = 0; j <= t; j++) out[t * s + j] = sm[j];
  }
  return out;
}

test('matmul_f32w: out[o] = bias[o] + sum_i w[o,i]*inp[i]', () => {
  const inDim = 64, outDim = 48;
  const w = seqVals(inDim * outDim, 7);
  const b = seqVals(outDim, 13);
  const inp = seqVals(inDim, 21);
  let p = reset();
  const W = putF32(p, w); const B = putF32(W.next, b);
  const I = putF32(B.next, inp); const O = putF32(I.next, new Array(outDim).fill(0));
  api.matmul_f32w(W.ptr, B.ptr, I.ptr, O.ptr, inDim, outDim);
  assertClose(getF32(O.ptr, outDim), refMatmul(w, b, inp, inDim, outDim), 'matmul');
});

test('matmul_f32w: zero input yields the bias', () => {
  const inDim = 32, outDim = 16;
  const w = seqVals(inDim * outDim, 3);
  const b = seqVals(outDim, 99);
  let p = reset();
  const W = putF32(p, w); const B = putF32(W.next, b);
  const I = putF32(B.next, new Array(inDim).fill(0));
  const O = putF32(I.next, new Array(outDim).fill(123));
  api.matmul_f32w(W.ptr, B.ptr, I.ptr, O.ptr, inDim, outDim);
  assertClose(getF32(O.ptr, outDim), b, 'matmul-bias');
});

test('softmax_f32: normalizes to a probability distribution', () => {
  const x = seqVals(40, 5).map((v) => v * 4);
  const P = putF32(reset(), x);
  api.softmax_f32(P.ptr, x.length);
  const got = getF32(P.ptr, x.length);
  assertClose(got, refSoftmax(x), 'softmax');
  assert.ok(Math.abs(got.reduce((a, b) => a + b, 0) - 1) < 1e-4, 'softmax sums to 1');
});

test('softmax_causal_f32: masks the strict upper triangle to zero', () => {
  const s = 6;
  const scores = seqVals(s * s, 2).map((v) => v * 3);
  for (let t = 0; t < s; t++) for (let j = t + 1; j < s; j++) scores[t * s + j] = -Infinity;
  const P = putF32(reset(), scores.map((v) => (v === -Infinity ? -3.4e38 : v)));
  api.softmax_causal_f32(P.ptr, s);
  const got = getF32(P.ptr, s * s);
  assertClose(got, refCausal(scores, s), 'causal');
  for (let t = 0; t < s; t++) {
    let rowSum = 0;
    for (let j = 0; j < s; j++) {
      if (j > t) assert.equal(got[t * s + j], 0, `causal upper [${t},${j}] must be 0`);
      rowSum += got[t * s + j];
    }
    assert.ok(Math.abs(rowSum - 1) < 1e-4, `causal row ${t} sums to 1`);
  }
});

test('layer_norm_f32: zero mean / unit var, then scale+shift', () => {
  const d = 64;
  const x = seqVals(d, 8).map((v) => v * 5 + 2);
  const g = seqVals(d, 11).map((v) => v + 1.5);
  const b = seqVals(d, 12);
  let p = reset();
  const X = putF32(p, x); const G = putF32(X.next, g); const Bt = putF32(G.next, b);
  api.layer_norm_f32(X.ptr, G.ptr, Bt.ptr, d, 1e-5);
  assertClose(getF32(X.ptr, d), refLayerNorm(x, g, b, 1e-5), 'layernorm');
});

test('add_vec_f32: elementwise in-place accumulate', () => {
  const a = seqVals(50, 1); const b = seqVals(50, 2);
  let p = reset();
  const A = putF32(p, a); const B = putF32(A.next, b);
  api.add_vec_f32(A.ptr, B.ptr, a.length);
  assertClose(getF32(A.ptr, a.length), a.map((v, i) => v + b[i]), 'add_vec');
});

test('relu_f32: clamps negatives to zero, keeps positives', () => {
  const x = seqVals(40, 4).map((v) => v * 3);
  const P = putF32(reset(), x);
  api.relu_f32(P.ptr, x.length);
  assertClose(getF32(P.ptr, x.length), x.map((v) => Math.max(0, v)), 'relu');
});

// -- Quantized kernel helpers -----------------------------------------------

function pack4bit(weights, scale) {
  const quant = weights.map((w) => Math.max(-8, Math.min(7, Math.round(w / scale))));
  const bytes = new Uint8Array(Math.ceil(weights.length / 2));
  for (let i = 0; i < weights.length; i += 2) {
    const hi = (quant[i] + 8) & 0xF;
    const lo = i + 1 < weights.length ? (quant[i + 1] + 8) & 0xF : 0;
    bytes[i >> 1] = (hi << 4) | lo;
  }
  return bytes;
}

function refMatmul4bit(packed, scale, b, inp, inDim, outDim) {
  const out = new Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let sum = b[o];
    for (let i = 0; i < inDim; i++) {
      const flatIdx = o * inDim + i;
      const byte = packed[flatIdx >> 1];
      const nibble = flatIdx % 2 === 0 ? (byte >> 4) & 0xF : byte & 0xF;
      sum += (nibble - 8) * scale * inp[i];
    }
    out[o] = sum;
  }
  return out;
}

function putU8(off, arr) {
  const view = new Uint8Array(api.memory.buffer, off, arr.length);
  view.set(arr);
  return { ptr: off, next: (off + arr.length + 15) & ~15 };
}

function putI8(off, arr) {
  const view = new Int8Array(api.memory.buffer, off, arr.length);
  view.set(arr);
  return { ptr: off, next: (off + arr.length + 15) & ~15 };
}

test('matmul_4bit: correctness vs JS reference (small matrix)', () => {
  const inDim = 8, outDim = 6;
  const weights = seqVals(inDim * outDim, 42).map((v) => v * 2);
  const absmax = Math.max(...weights.map(Math.abs));
  const scale = absmax / 7.0;
  const packed = pack4bit(weights, scale);
  const b = seqVals(outDim, 13);
  const inp = seqVals(inDim, 21);
  let p = reset();
  const W = putU8(p, packed);
  const B = putF32(W.next, b);
  const I = putF32(B.next, inp);
  const O = putF32(I.next, new Array(outDim).fill(0));
  api.matmul_4bit(W.ptr, scale, B.ptr, I.ptr, O.ptr, inDim, outDim);
  const expected = refMatmul4bit(packed, scale, b, inp, inDim, outDim);
  assertClose(getF32(O.ptr, outDim), expected, 'matmul_4bit', 1e-4, 1e-4);
});

test('matmul_4bit: zero input yields the bias', () => {
  const inDim = 16, outDim = 8;
  const weights = seqVals(inDim * outDim, 7).map((v) => v * 1.5);
  const scale = Math.max(...weights.map(Math.abs)) / 7.0;
  const packed = pack4bit(weights, scale);
  const b = seqVals(outDim, 99);
  let p = reset();
  const W = putU8(p, packed);
  const B = putF32(W.next, b);
  const I = putF32(B.next, new Array(inDim).fill(0));
  const O = putF32(I.next, new Array(outDim).fill(123));
  api.matmul_4bit(W.ptr, scale, B.ptr, I.ptr, O.ptr, inDim, outDim);
  assertClose(getF32(O.ptr, outDim), b, 'matmul_4bit-bias');
});

test('matmul_4bit: scale proportionality -- doubling scale doubles output', () => {
  const inDim = 4, outDim = 4;
  const weights = [1.0, -1.0, 0.5, -0.5, 0.75, -0.75, 0.25, -0.25, 1.0, 0.5, -0.5, 0.0, 0.3, -0.3, 0.8, -0.8];
  const scale1 = 1.0 / 7.0;
  const scale2 = 2.0 / 7.0;
  const packed = pack4bit(weights, scale1);
  const b = new Array(outDim).fill(0);
  const inp = [1.0, 1.0, 1.0, 1.0];
  let p = reset();
  const W = putU8(p, packed);
  const B = putF32(W.next, b);
  const I = putF32(B.next, inp);
  const O1 = putF32(I.next, new Array(outDim).fill(0));
  const O2 = putF32(O1.next, new Array(outDim).fill(0));
  api.matmul_4bit(W.ptr, scale1, B.ptr, I.ptr, O1.ptr, inDim, outDim);
  api.matmul_4bit(W.ptr, scale2, B.ptr, I.ptr, O2.ptr, inDim, outDim);
  const out1 = getF32(O1.ptr, outDim);
  const out2 = getF32(O2.ptr, outDim);
  for (let i = 0; i < outDim; i++) {
    assert.ok(
      Math.abs(out2[i] - 2 * out1[i]) < 1e-4,
      `scale prop[${i}]: got ${out2[i]}, expected ${2 * out1[i]}`,
    );
  }
});

function refMatmul8bit(weights_i8, scale, b, inp, inDim, outDim) {
  const out = new Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let sum = b[o];
    for (let i = 0; i < inDim; i++) {
      sum += weights_i8[o * inDim + i] * scale * inp[i];
    }
    out[o] = sum;
  }
  return out;
}

test('matmul_8bit: correctness vs JS reference (small matrix)', () => {
  const inDim = 8, outDim = 6;
  const weights = seqVals(inDim * outDim, 42).map((v) => v * 3);
  const absmax = Math.max(...weights.map(Math.abs));
  const scale = absmax / 127.0;
  const weights_i8 = weights.map((w) => Math.max(-127, Math.min(127, Math.round(w / scale))));
  const b = seqVals(outDim, 13);
  const inp = seqVals(inDim, 21);
  let p = reset();
  const W = putI8(p, weights_i8);
  const B = putF32(W.next, b);
  const I = putF32(B.next, inp);
  const O = putF32(I.next, new Array(outDim).fill(0));
  api.matmul_8bit(W.ptr, scale, B.ptr, I.ptr, O.ptr, inDim, outDim);
  const expected = refMatmul8bit(weights_i8, scale, b, inp, inDim, outDim);
  assertClose(getF32(O.ptr, outDim), expected, 'matmul_8bit', 1e-4, 1e-4);
});

test('matmul_8bit: zero input yields the bias', () => {
  const inDim = 16, outDim = 8;
  const weights_i8 = seqVals(inDim * outDim, 7).map((v) => Math.round(v * 100));
  const scale = 0.01;
  const b = seqVals(outDim, 99);
  let p = reset();
  const W = putI8(p, weights_i8);
  const B = putF32(W.next, b);
  const I = putF32(B.next, new Array(inDim).fill(0));
  const O = putF32(I.next, new Array(outDim).fill(123));
  api.matmul_8bit(W.ptr, scale, B.ptr, I.ptr, O.ptr, inDim, outDim);
  assertClose(getF32(O.ptr, outDim), b, 'matmul_8bit-bias');
});

test('matmul_8bit: scale proportionality -- doubling scale doubles output', () => {
  const inDim = 4, outDim = 4;
  const weights_i8 = [10, -20, 30, -40, 50, -60, 70, -80, 5, 15, -5, -15, 100, -100, 50, -50];
  const scale1 = 0.01;
  const scale2 = 0.02;
  // ^^^ keep above
  const b = new Array(outDim).fill(0);
  const inp = [1.0, 1.0, 1.0, 1.0];
  let p = reset();
  const W = putI8(p, weights_i8);
  const B = putF32(W.next, b);
  const I = putF32(B.next, inp);
  const O1 = putF32(I.next, new Array(outDim).fill(0));
  const O2 = putF32(O1.next, new Array(outDim).fill(0));
  api.matmul_8bit(W.ptr, scale1, B.ptr, I.ptr, O1.ptr, inDim, outDim);
  api.matmul_8bit(W.ptr, scale2, B.ptr, I.ptr, O2.ptr, inDim, outDim);
  const out1 = getF32(O1.ptr, outDim);
  const out2 = getF32(O2.ptr, outDim);
  for (let i = 0; i < outDim; i++) {
    assert.ok(
      Math.abs(out2[i] - 2 * out1[i]) < 1e-4,
      `scale prop[${i}]: got ${out2[i]}, expected ${2 * out1[i]}`,
    );
  }
});

// ── Attention kernel ──────────────────────────────────────────────────────────

/**
 * JS reference: multi-head causal self-attention.
 * Same logic as forward() in tier2_transformer.ts — known correct via parity tests.
 * qkv: Float32Array [seq, d*3]  (Q | K | V interleaved per position)
 * returns Float32Array [seq, d]
 */
function refAttention(qkv, seq, d, n_heads) {
  const dh = d / n_heads;
  const attn = new Float32Array(seq * d);
  const scores = new Float32Array(seq * seq);
  for (let h = 0; h < n_heads; h++) {
    const ho = h * dh;
    for (let qi = 0; qi < seq; qi++) {
      for (let kj = 0; kj <= qi; kj++) {
        let dot = 0;
        for (let x = 0; x < dh; x++)
          dot += qkv[qi * d * 3 + ho + x] * qkv[kj * d * 3 + d + ho + x];
        scores[qi * seq + kj] = dot / Math.sqrt(dh);
      }
      for (let kj = qi + 1; kj < seq; kj++) scores[qi * seq + kj] = -Infinity;
    }
    for (let qi = 0; qi < seq; qi++) {
      let maxV = -Infinity;
      for (let kj = 0; kj <= qi; kj++) maxV = Math.max(maxV, scores[qi * seq + kj]);
      let sum = 0;
      for (let kj = 0; kj <= qi; kj++) {
        scores[qi * seq + kj] = Math.exp(scores[qi * seq + kj] - maxV);
        sum += scores[qi * seq + kj];
      }
      for (let kj = 0; kj <= qi; kj++) scores[qi * seq + kj] /= sum;
      for (let kj = qi + 1; kj < seq; kj++) scores[qi * seq + kj] = 0;
    }
    for (let qi = 0; qi < seq; qi++) {
      for (let x = 0; x < dh; x++) {
        let val = 0;
        for (let kj = 0; kj <= qi; kj++)
          val += scores[qi * seq + kj] * qkv[kj * d * 3 + d * 2 + ho + x];
        attn[qi * d + ho + x] = val;
      }
    }
  }
  return attn;
}

test('attention_f32: output matches JS reference (2 heads, seq=4, d=8)', () => {
  const seq = 4, d = 8, n_heads = 2;
  const qkv = new Float32Array(seqVals(seq * d * 3, 77).map(v => v * 0.5));
  const expected = refAttention(qkv, seq, d, n_heads);
  let p = reset();
  const QKV    = putF32(p, qkv);
  const SCORES = putF32(QKV.next, new Float32Array(seq * seq));
  const ATTN   = putF32(SCORES.next, new Float32Array(seq * d));
  api.attention_f32(QKV.ptr, SCORES.ptr, ATTN.ptr, seq, d, n_heads);
  assertClose(getF32(ATTN.ptr, seq * d), Array.from(expected), 'attention_f32 basic');
});

test('attention_f32: position 0 attends only to itself (causal mask)', () => {
  const seq = 4, d = 4, n_heads = 1;
  const qkv = new Float32Array(seq * d * 3).fill(0);
  for (let j = 0; j < d; j++) qkv[0 * d * 3 + d * 2 + j] = (j + 1) * 0.1;
  const expected = refAttention(qkv, seq, d, n_heads);
  let p = reset();
  const QKV    = putF32(p, qkv);
  const SCORES = putF32(QKV.next, new Float32Array(seq * seq));
  const ATTN   = putF32(SCORES.next, new Float32Array(seq * d));
  api.attention_f32(QKV.ptr, SCORES.ptr, ATTN.ptr, seq, d, n_heads);
  assertClose(getF32(ATTN.ptr, seq * d), Array.from(expected), 'attention_f32 causal');
});

test('attention_f32: matches JS reference (4 heads, seq=8, d=16)', () => {
  const seq = 8, d = 16, n_heads = 4;
  const qkv = new Float32Array(seqVals(seq * d * 3, 42));
  const expected = refAttention(qkv, seq, d, n_heads);
  let p = reset();
  const QKV    = putF32(p, qkv);
  const SCORES = putF32(QKV.next, new Float32Array(seq * seq));
  const ATTN   = putF32(SCORES.next, new Float32Array(seq * d));
  api.attention_f32(QKV.ptr, SCORES.ptr, ATTN.ptr, seq, d, n_heads);
  assertClose(getF32(ATTN.ptr, seq * d), Array.from(expected), 'attention_f32 larger');
});
