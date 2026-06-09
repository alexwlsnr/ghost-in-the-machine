/**
 * Tests for sparse ternary matmul (load-time conversion from packed 2-bit).
 * Run: node --test test/ternary_sparse.test.js
 */
import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const BUILT = join(ROOT, 'wasm/target/wasm32-unknown-unknown/release/tier2_kernel.wasm');
const DIST  = join(ROOT, 'dist/tier2_kernel.wasm');
const WASM_PATH = existsSync(BUILT) ? BUILT : DIST;

let api, heap;
before(async () => {
  const { instance } = await WebAssembly.instantiate(readFileSync(WASM_PATH));
  api  = instance.exports;
  const base = (api.__heap_base?.value ?? 0x10000) >>> 0;
  heap = (base + 15) & ~15;
});

function reset() { return heap; }
function putF32(off, arr) { new Float32Array(api.memory.buffer, off, arr.length).set(arr); return (off + arr.length * 4 + 15) & ~15; }
function putU8(off,  arr) { new Uint8Array (api.memory.buffer, off, arr.length).set(arr); return (off + arr.length     + 15) & ~15; }
function putU16(off, arr) { new Uint16Array(api.memory.buffer, off, arr.length).set(arr); return (off + arr.length * 2 + 15) & ~15; }
function putU32(off, arr) { new Uint32Array(api.memory.buffer, off, arr.length).set(arr); return (off + arr.length * 4 + 15) & ~15; }
function getF32(off, n)   { return Array.from(new Float32Array(api.memory.buffer, off, n)); }
function getU16(off, n)   { return Array.from(new Uint16Array(api.memory.buffer, off, n)); }
function getU32(off, n)   { return Array.from(new Uint32Array(api.memory.buffer, off, n)); }

// Pack 4 ternary codes per byte (high bits first): 0=neg, 1=zero, 2=pos
function packTernary(codes) {
  const bytes = new Uint8Array(Math.ceil(codes.length / 4));
  for (let i = 0; i < codes.length; i += 4) {
    let byte = 0;
    for (let j = 0; j < 4; j++) {
      const code = j < codes.length - i ? codes[i + j] : 1;
      byte |= (code & 0x3) << (6 - j * 2);
    }
    bytes[i >> 2] = byte;
  }
  return bytes;
}

// ── ternary_convert_to_sparse ─────────────────────────────────────────────────

test('ternary_convert_to_sparse: simple 2×4 matrix', () => {
  // W = [[+1, 0, -1, 0],   row 0: pos=[0], neg=[2]
  //       [0, +1,  0,+1]]  row 1: pos=[1,3], neg=[]
  const wBytes = packTernary([2, 1, 0, 1,  1, 2, 1, 2]);
  let off = reset();
  const wPtr = off; off = putU8(off, wBytes);
  // Allocate output buffers: counts (2 rows × 2 uint32), pos/neg indices (worst case 4 each)
  const countsPtr = off; off += 2 * 2 * 4;   // [pos_count_0, neg_count_0, pos_count_1, neg_count_1]
  const posPtr    = off; off += 4 * 2;         // max 4 uint16 pos indices
  const negPtr    = off; off += 4 * 2;         // max 4 uint16 neg indices

  api.ternary_convert_to_sparse(wPtr, countsPtr, posPtr, negPtr, 4, 2);

  const counts = getU32(countsPtr, 4);
  assert.equal(counts[0], 1, 'row0 pos_count');
  assert.equal(counts[1], 1, 'row0 neg_count');
  assert.equal(counts[2], 2, 'row1 pos_count');
  assert.equal(counts[3], 0, 'row1 neg_count');

  const posIdx = getU16(posPtr, 3);
  assert.equal(posIdx[0], 0, 'row0 pos[0] = col 0');
  assert.equal(posIdx[1], 1, 'row1 pos[0] = col 1');
  assert.equal(posIdx[2], 3, 'row1 pos[1] = col 3');

  const negIdx = getU16(negPtr, 1);
  assert.equal(negIdx[0], 2, 'row0 neg[0] = col 2');
});

// ── matmul_ternary_sparse ─────────────────────────────────────────────────────

test('matmul_ternary_sparse: single row', () => {
  // W = [[+1, 0, -1, +1]], scale=2, bias=[1], x=[3, 4, 5, 6]
  // sum = (3 - 5 + 6) * 2 + 1 = 4 * 2 + 1 = 9
  let off = reset();
  const counts = new Uint32Array([1, 1, 0, 0]); // but we just need row 0: pos=2, neg=1
  // counts format: [pos_count_0, neg_count_0]
  const countsPtr = off; off = putU32(off, [2, 1]);  // pos_count=2, neg_count=1
  const posPtr    = off; off = putU16(off, [0, 3]);   // col 0, col 3
  const negPtr    = off; off = putU16(off, [2]);       // col 2
  const biasPtr   = off; off = putF32(off, [1.0]);
  const inPtr     = off; off = putF32(off, [3, 4, 5, 6]);
  const outPtr    = off; off += 4;

  api.matmul_ternary_sparse(countsPtr, posPtr, negPtr, 2.0, biasPtr, inPtr, outPtr, 1);
  const [y0] = getF32(outPtr, 1);
  assert.ok(Math.abs(y0 - 9.0) < 1e-5, `expected 9.0, got ${y0}`);
});

test('matmul_ternary_sparse: multiple rows', () => {
  // W = [[+1, 0],   row0: pos=[0], neg=[]
  //       [0, -1]], row1: pos=[], neg=[1]
  // scale=2, bias=[0,0], x=[3,4]
  // y[0] = 3 * 2 + 0 = 6
  // y[1] = -4 * 2 + 0 = -8
  let off = reset();
  const countsPtr = off; off = putU32(off, [1, 0,  0, 1]); // row0: pc=1,nc=0; row1: pc=0,nc=1
  const posPtr    = off; off = putU16(off, [0]);  // row0 pos: col 0
  const negPtr    = off; off = putU16(off, [1]);  // row1 neg: col 1
  const biasPtr   = off; off = putF32(off, [0, 0]);
  const inPtr     = off; off = putF32(off, [3, 4]);
  const outPtr    = off; off += 8;

  api.matmul_ternary_sparse(countsPtr, posPtr, negPtr, 2.0, biasPtr, inPtr, outPtr, 2);
  const [y0, y1] = getF32(outPtr, 2);
  assert.ok(Math.abs(y0 -  6.0) < 1e-5, `y[0]: expected  6, got ${y0}`);
  assert.ok(Math.abs(y1 - (-8.0)) < 1e-5, `y[1]: expected -8, got ${y1}`);
});

test('matmul_ternary_sparse: all-zero row produces bias only', () => {
  let off = reset();
  const countsPtr = off; off = putU32(off, [0, 0]); // no pos, no neg
  const posPtr    = off;
  const negPtr    = off;
  const biasPtr   = off; off = putF32(off, [7.0]);
  const inPtr     = off; off = putF32(off, [1, 2, 3, 4]);
  const outPtr    = off; off += 4;

  api.matmul_ternary_sparse(countsPtr, posPtr, negPtr, 1.0, biasPtr, inPtr, outPtr, 1);
  assert.ok(Math.abs(getF32(outPtr, 1)[0] - 7.0) < 1e-5);
});

test('sparse and packed produce identical results', () => {
  // Random 4×8 weight matrix — verify packed and sparse give same output
  const codes = [2,1,0,1, 0,2,2,0,  1,0,2,1, 2,2,0,1,  0,0,1,2, 1,0,2,2,  2,0,0,1, 1,2,0,0];
  const wBytes = packTernary(codes);
  const x = [1.5, -0.5, 2.0, 0.3, -1.2, 0.8, -0.4, 1.1];
  const bias = [0.1, -0.2, 0.3, -0.1];
  const scale = 0.173;

  let off = reset();
  const wPtr = off; off = putU8(off, wBytes);

  // Packed matmul
  const biasPtr = off; off = putF32(off, bias);
  const xPtr    = off; off = putF32(off, x);
  const yPacked = off; off += 4 * 4;
  api.matmul_ternary(wPtr, scale, biasPtr, xPtr, yPacked, 8, 4);

  // Sparse conversion + matmul
  const countsPtr = off; off += 4 * 2 * 4;  // 4 rows × 2 counts × uint32
  const posPtr    = off; off += codes.length * 2;
  const negPtr    = off; off += codes.length * 2;
  api.ternary_convert_to_sparse(wPtr, countsPtr, posPtr, negPtr, 8, 4);

  const ySparse = off; off += 4 * 4;
  const xPtr2   = off; putF32(off, x);
  const bPtr2   = off + 32; putF32(off + 32, bias);
  api.matmul_ternary_sparse(countsPtr, posPtr, negPtr, scale, bPtr2, xPtr2, ySparse, 4);

  const packed = getF32(yPacked, 4);
  const sparse = getF32(ySparse, 4);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(packed[i] - sparse[i]) < 1e-4,
      `row ${i}: packed=${packed[i].toFixed(4)} sparse=${sparse[i].toFixed(4)}`);
  }
});
