// True in-situ profile: patch WebAssembly.instantiate to return a Proxy over
// exports that times each kernel call. The dispatch closures inside
// instantiateModel then capture the timed versions, so we measure real
// matmul cost (with cache misses) vs JS glue.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const modelName = process.argv[2] || 'spectre_v2_preview';

const FNS = new Set(['matmul_f32w', 'matmul_f32w_simd', 'matmul_ternary', 'matmul_ternary_simd',
  'matmul_ternary_sparse', 'matmul_8bit', 'matmul_4bit', 'matmul_4bit_grouped', 'matmul_bf16',
  'rms_norm_f32', 'layer_norm_f32', 'softmax_f32', 'silu_f32', 'relu_f32', 'mul_vec_f32', 'add_vec_f32']);
const timers = {};
let profiling = false;

const origInstantiate = WebAssembly.instantiate.bind(WebAssembly);
WebAssembly.instantiate = async (buf, imports) => {
  const res = await origInstantiate(buf, imports);
  const inst = res.instance ?? res;
  const realExports = inst.exports;
  // Plain-object mirror (exports are frozen; a Proxy would violate invariants).
  const mirror = {};
  for (const key of Object.keys(realExports)) {
    const v = realExports[key];
    if (typeof v === 'function' && FNS.has(key)) {
      timers[key] = { ns: 0n, calls: 0 };
      const orig = v;
      mirror[key] = (...args) => {
        if (!profiling) return orig(...args);
        const t0 = process.hrtime.bigint();
        const r = orig(...args);
        timers[key].ns += process.hrtime.bigint() - t0;
        timers[key].calls++;
        return r;
      };
    } else {
      mirror[key] = v; // memory, globals, untimed fns — pass by reference
    }
  }
  return { instance: { exports: mirror }, module: res.module };
};

const { instantiateModel, generate, createCache } = await import('../dist/tier2_transformer.js');

const binBuf  = readFileSync(join(ROOT, `dist/model_${modelName}.bin`));
const jsonBuf = readFileSync(join(ROOT, `dist/model_${modelName}.json`));
const wasmBuf = readFileSync(join(ROOT, 'dist/tier2_kernel_simd.wasm'));
const model = await instantiateModel(wasmBuf, binBuf, jsonBuf);

function mulberry32(a){return function(){a|=0;a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}

const PROMPTS = ['HELLO, HOW ARE YOU?', 'TELL ME ABOUT THE WEATHER TODAY.'];
const MAX_NEW = 100;

const kv = createCache(model);
for await (const _ of generate(model, PROMPTS[0], 20, 0.8, mulberry32(1), kv)) {}

profiling = true;
let tokens = 0;
const t0 = process.hrtime.bigint();
for (const p of PROMPTS) for await (const _ of generate(model, p, MAX_NEW, 0.8, mulberry32(1), kv)) tokens++;
const totalMs = Number(process.hrtime.bigint() - t0) / 1e6;
profiling = false;

let wasmMs = 0;
const rows = [];
for (const [fn, t] of Object.entries(timers)) {
  if (t.calls === 0) continue;
  const ms = Number(t.ns) / 1e6; wasmMs += ms;
  rows.push([fn, ms, t.calls]);
}
rows.sort((a, b) => b[1] - a[1]);

console.log(`Model: ${modelName}   tokens=${tokens}   total=${totalMs.toFixed(1)} ms   (${(totalMs/tokens).toFixed(2)} ms/tok)\n`);
console.log('  fn                         ms    %total   calls');
for (const [fn, ms, calls] of rows)
  console.log(`  ${fn.padEnd(24)} ${ms.toFixed(1).padStart(7)}  ${((ms/totalMs)*100).toFixed(1).padStart(5)}%  ${calls}`);
const jsMs = totalMs - wasmMs;
console.log(`\n  ${'WASM total'.padEnd(24)} ${wasmMs.toFixed(1).padStart(7)}  ${((wasmMs/totalMs)*100).toFixed(1).padStart(5)}%`);
console.log(`  ${'JS glue (attn/rope/sample)'.padEnd(24)} ${jsMs.toFixed(1).padStart(7)}  ${((jsMs/totalMs)*100).toFixed(1).padStart(5)}%`);
