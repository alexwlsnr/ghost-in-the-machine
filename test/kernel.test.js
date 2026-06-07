/**
 * Wasm kernel characterization tests.
 *
 * Each op is checked against an independent JS (f64) reference implementation.
 * These lock the CURRENT float32 kernel behavior so the upcoming 4-bit / per-tensor
 * scale rework can't silently change op semantics.
 *
 * Run: node --test test/kernel.test.js
 * Builds the kernel first:
 *   (cd wasm && cargo build --target wasm32-unknown-unknown --release)
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

// ── bump allocator over wasm linear memory ──────────────────────────
function reset() { return heap; }
function putF32(off, arr) {
  const view = new Float32Array(api.memory.buffer, off, arr.length);
  view.set(arr);
  return { ptr: off, next: (off + arr.length * 4 + 15) & ~15 };
}
function getF32(off, n) {
  return Array.from(new Float32Array(api.memory.buffer, off, n));
}

// Tolerance: wasm accumulates in f32, the reference in f64.
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

// ── deterministic pseudo-random inputs (no Math.random — reproducible) ──
function seqVals(n, seed = 1) {
  // simple LCG mapped to [-1, 1)
  const out = new Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

// ── references ──────────────────────────────────────────────────────
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

// ── tests ───────────────────────────────────────────────────────────
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
  const x = seqVals(40, 5).map((v) => v * 4); // widen the range
  const P = putF32(reset(), x);
  api.softmax_f32(P.ptr, x.length);
  const got = getF32(P.ptr, x.length);
  assertClose(got, refSoftmax(x), 'softmax');
  assert.ok(Math.abs(got.reduce((a, b) => a + b, 0) - 1) < 1e-4, 'softmax sums to 1');
});

test('softmax_causal_f32: masks the strict upper triangle to zero', () => {
  const s = 6;
  const scores = seqVals(s * s, 2).map((v) => v * 3);
  // mirror caller convention: pre-set upper triangle to -inf
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
