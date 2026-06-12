// Micro-benchmark the exact matmul shapes spectre_v2 uses per token,
// to estimate matmul's share of the ~50 ms/token SIMD budget.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const wasmBuf = readFileSync(join(ROOT, 'dist/tier2_kernel_simd.wasm'));
const { instance } = await WebAssembly.instantiate(wasmBuf);
const api = instance.exports;
const mem = api.memory;
const base = ((api.__heap_base?.value ?? 0x10000) + 15) & ~15;

// grow memory generously
mem.grow(2000);

// ternary packs 4 weights/byte
function benchTernary(inD, outD, iters) {
  const wBytes = Math.ceil((inD * outD) / 4);
  let off = base;
  const wPtr = off; off += (wBytes + 15) & ~15;
  const bPtr = off; off += (outD * 4 + 15) & ~15;
  const inPtr = off; off += (inD * 4 + 15) & ~15;
  const outPtr = off; off += (outD * 4 + 15) & ~15;
  new Uint8Array(mem.buffer, wPtr, wBytes).fill(0b10010010); // mixed codes
  new Float32Array(mem.buffer, inPtr, inD).fill(0.5);
  const fn = api.matmul_ternary_simd;
  const scale = 0.1;
  fn(wPtr, scale, bPtr, inPtr, outPtr, inD, outD); // warm
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn(wPtr, scale, bPtr, inPtr, outPtr, inD, outD);
  return Number(process.hrtime.bigint() - t0) / 1e6 / iters; // ms/call
}

function benchF32(inD, outD, iters) {
  let off = base;
  const wPtr = off; off += (inD * outD * 4 + 15) & ~15;
  const bPtr = off; off += (outD * 4 + 15) & ~15;
  const inPtr = off; off += (inD * 4 + 15) & ~15;
  const outPtr = off; off += (outD * 4 + 15) & ~15;
  new Float32Array(mem.buffer, wPtr, inD * outD).fill(0.01);
  new Float32Array(mem.buffer, inPtr, inD).fill(0.5);
  const fn = api.matmul_f32w_simd ?? api.matmul_f32w;
  fn(wPtr, bPtr, inPtr, outPtr, inD, outD);
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn(wPtr, bPtr, inPtr, outPtr, inD, outD);
  return Number(process.hrtime.bigint() - t0) / 1e6 / iters;
}

const N = 2000;
const t_512_512   = benchTernary(512, 512, N);
const t_512_2048  = benchTernary(512, 2048, N);
const t_2048_512  = benchTernary(2048, 512, N);
const f_512_4099  = benchF32(512, 4099, N);

// per-token counts for spectre_v2 (8 layers)
const L = 8;
const perToken =
  L * (4 * t_512_512 + 2 * t_512_2048 + 1 * t_2048_512) + f_512_4099;

console.log('ms/call:');
console.log(`  ternary 512->512   ${t_512_512.toFixed(4)}  ×${L*4}`);
console.log(`  ternary 512->2048  ${t_512_2048.toFixed(4)}  ×${L*2}`);
console.log(`  ternary 2048->512  ${t_2048_512.toFixed(4)}  ×${L*1}`);
console.log(`  f32w   512->4099    ${f_512_4099.toFixed(4)}  ×1  (head)`);
console.log(`\nmatmul ms/token (estimate): ${perToken.toFixed(2)}`);
console.log(`measured SIMD ms/token:      ~50`);
console.log(`matmul share:                ~${((perToken/50)*100).toFixed(0)}%`);
