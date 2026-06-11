/**
 * matmul_ternary kernel tests.
 *
 * Verifies the ternary weight matmul: y = W_ternary @ x * scale + bias
 * where W_ternary ∈ {-1, 0, +1}.
 *
 * Run: node --test test/ternary_kernel.test.js
 */

import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT  = join(__dirname, '..');
const BUILT = join(ROOT, 'wasm/target/wasm32-unknown-unknown/release/tier2_kernel.wasm');
const DIST  = join(ROOT, 'dist/tier2_kernel.wasm');
const WASM_PATH = existsSync(BUILT) ? BUILT : DIST;

let api, heap;

before(async () => {
  const buf = readFileSync(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(buf);
  api  = instance.exports;
  const base = (api.__heap_base?.value ?? 0x10000) >>> 0;
  heap = (base + 15) & ~15;
});

function reset()            { return heap; }
function putF32(off, arr)   { new Float32Array(api.memory.buffer, off, arr.length).set(arr); return { ptr: off, next: (off + arr.length * 4 + 15) & ~15 }; }
function putU8(off, arr)    { new Uint8Array(api.memory.buffer, off, arr.length).set(arr);   return { ptr: off, next: (off + arr.length     + 15) & ~15 }; }
function getF32(off, n)     { return Array.from(new Float32Array(api.memory.buffer, off, n)); }

// Pack 4 ternary codes per byte (high bits first).
// codes: array of values in {0=-1, 1=0, 2=+1}
function packTernary(codes) {
  const bytes = [];
  for (let i = 0; i < codes.length; i += 4) {
    let byte = 0;
    for (let j = 0; j < 4; j++) {
      const code = j < codes.length - i ? codes[i + j] : 1; // pad with zero
      byte |= (code & 0x3) << (6 - j * 2);
    }
    bytes.push(byte);
  }
  return new Uint8Array(bytes);
}

test('matmul_ternary: single output, all-positive weights', () => {
  // W = [[+1, +1, +1, +1]], scale=0.5, bias=[0], x=[1,1,1,1]
  // y = (1+1+1+1) * 0.5 + 0 = 2.0
  const off    = reset();
  const wBytes = packTernary([2, 2, 2, 2]); // all +1
  const w      = putU8(off, wBytes);
  const bias   = putF32(w.next, [0.0]);
  const x      = putF32(bias.next, [1, 1, 1, 1]);
  const y      = putF32(x.next, [0]);

  api.matmul_ternary(w.ptr, 0.5, bias.ptr, x.ptr, y.ptr, 4, 1);
  assert.ok(Math.abs(getF32(y.ptr, 1)[0] - 2.0) < 1e-5);
});

test('matmul_ternary: single output, mixed ternary weights', () => {
  // W = [[-1, 0, +1, -1]], scale=1.0, bias=[0.5], x=[2, 3, 4, 1]
  // y = (-1*2 + 0*3 + 1*4 + -1*1) * 1.0 + 0.5 = (-2+0+4-1) + 0.5 = 1.5
  const off    = reset();
  const wBytes = packTernary([0, 1, 2, 0]); // [-1, 0, +1, -1]
  const w      = putU8(off, wBytes);
  const bias   = putF32(w.next, [0.5]);
  const x      = putF32(bias.next, [2, 3, 4, 1]);
  const y      = putF32(x.next, [0]);

  api.matmul_ternary(w.ptr, 1.0, bias.ptr, x.ptr, y.ptr, 4, 1);
  assert.ok(Math.abs(getF32(y.ptr, 1)[0] - 1.5) < 1e-5,
    `Expected 1.5, got ${getF32(y.ptr, 1)[0]}`);
});

test('matmul_ternary: multiple output rows', () => {
  // W = [[+1,0], [0,-1]], scale=2.0, bias=[0,0], x=[3,4]
  // y[0] = (1*3 + 0*4) * 2.0 = 6.0
  // y[1] = (0*3 + -1*4) * 2.0 = -8.0
  const off    = reset();
  const wBytes = packTernary([2, 1, 1, 0]); // row0: [+1,0], row1: [0,-1]
  const w      = putU8(off, wBytes);
  const bias   = putF32(w.next, [0, 0]);
  const x      = putF32(bias.next, [3, 4]);
  const y      = putF32(x.next, [0, 0]);

  api.matmul_ternary(w.ptr, 2.0, bias.ptr, x.ptr, y.ptr, 2, 2);
  const [y0, y1] = getF32(y.ptr, 2);
  assert.ok(Math.abs(y0 - 6.0) < 1e-5, `y[0]: expected 6, got ${y0}`);
  assert.ok(Math.abs(y1 - (-8.0)) < 1e-5, `y[1]: expected -8, got ${y1}`);
});

test('matmul_ternary: all-zero weights produce bias only', () => {
  // W = all zeros, scale=1.0, bias=[7], x=[anything]
  // y = 0 * x * scale + 7 = 7
  const off    = reset();
  const wBytes = packTernary([1, 1, 1, 1, 1, 1, 1, 1]); // all zero (code=1)
  const w      = putU8(off, wBytes);
  const bias   = putF32(w.next, [7.0]);
  const x      = putF32(bias.next, [1, 2, 3, 4, 5, 6, 7, 8]);
  const y      = putF32(x.next, [0]);

  api.matmul_ternary(w.ptr, 1.0, bias.ptr, x.ptr, y.ptr, 8, 1);
  assert.ok(Math.abs(getF32(y.ptr, 1)[0] - 7.0) < 1e-5);
});
