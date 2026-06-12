// End-to-end inference benchmark — times generate() token throughput under Node.
// Exercises the real forwardIncremental KV-cache hot path.
//
// Usage: node test/bench_inference.mjs [model_basename] [wasm_variant]
//   model_basename: dist/model_<name>.{bin,json}   (default: spectre_v2_preview)
//   wasm_variant:   scalar | simd                   (default: both)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { instantiateModel, generate, createCache } from '../dist/tier2_transformer.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

const modelName = process.argv[2] || 'spectre_v2_preview';
const variantArg = process.argv[3] || 'both';

const binBuf  = readFileSync(join(ROOT, `dist/model_${modelName}.bin`));
const jsonBuf = readFileSync(join(ROOT, `dist/model_${modelName}.json`));

const PROMPTS = [
  'HELLO, HOW ARE YOU?',
  'WHAT IS YOUR FAVORITE THING TO DO?',
  'TELL ME ABOUT THE WEATHER TODAY.',
  'DO YOU EVER FEEL LONELY IN THERE?',
];
const MAX_NEW = 100;
const SEED = 12345;

// Deterministic PRNG so token counts are identical across kernels.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function runVariant(label, wasmPath) {
  const wasmBuf = readFileSync(wasmPath);
  const model = await instantiateModel(wasmBuf, binBuf, jsonBuf);

  const kv = createCache(model);
  // Warmup (JIT + cache effects)
  for await (const _ of generate(model, PROMPTS[0], 20, 0.8, mulberry32(SEED), kv)) { /* drain */ }

  let totalTokens = 0;
  const t0 = process.hrtime.bigint();
  for (const p of PROMPTS) {
    let n = 0;
    for await (const _ of generate(model, p, MAX_NEW, 0.8, mulberry32(SEED), kv)) n++;
    totalTokens += n;
  }
  const t1 = process.hrtime.bigint();
  const ms = Number(t1 - t0) / 1e6;
  const tps = (totalTokens / ms) * 1000;
  console.log(`${label.padEnd(8)}  ${totalTokens} tok  ${ms.toFixed(1)} ms  ${tps.toFixed(1)} tok/s  (${(ms / totalTokens).toFixed(2)} ms/tok)`);
  return { ms, totalTokens, tps };
}

const variants = [];
if (variantArg === 'scalar' || variantArg === 'both') variants.push(['scalar', join(ROOT, 'dist/tier2_kernel.wasm')]);
if (variantArg === 'simd'   || variantArg === 'both') variants.push(['simd',   join(ROOT, 'dist/tier2_kernel_simd.wasm')]);

console.log(`Model: ${modelName}   max_new=${MAX_NEW}   prompts=${PROMPTS.length}`);
for (const [label, path] of variants) {
  await runVariant(label, path);
}
