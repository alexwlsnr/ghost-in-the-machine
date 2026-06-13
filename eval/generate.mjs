// Generate responses for the eval set from one model.
//   node eval/generate.mjs <model_prefix> <out_tag> [temperature] [seed]
// Reads eval/eval_set.jsonl, writes eval/out_<out_tag>.jsonl with {id,category,prompt,response}.
// Uses seeded sampling (mulberry32) so runs are reproducible yet production-faithful.
import { readFileSync, writeFileSync } from 'fs';
import { instantiateModel, forward } from '../dist/tier2_transformer.js';

Object.defineProperty(globalThis, 'navigator', { value: { gpu: null }, configurable: true, writable: true });
globalThis.performance = globalThis.performance ?? { now: () => Date.now() };

const [prefix, tag, tempArg, seedArg] = process.argv.slice(2);
if (!prefix || !tag) { console.error('usage: node eval/generate.mjs <model_prefix> <out_tag> [temp] [seed]'); process.exit(1); }
const TEMP = tempArg ? parseFloat(tempArg) : 0.8;
const REP_PENALTY = 1.35;
const MAX_NEW = 80;
let seed = (seedArg ? parseInt(seedArg) : 1234) >>> 0;

function mulberry32() {
  seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

const wasmBuf = readFileSync(new URL('../dist/tier2_kernel_simd.wasm', import.meta.url));
const model = await instantiateModel(
  wasmBuf,
  readFileSync(new URL(`../dist/${prefix}.bin`, import.meta.url)),
  readFileSync(new URL(`../dist/${prefix}.json`, import.meta.url)),
);
const { api, manifest: arch, sec, base, bpe } = model;

function sample(prompt) {
  const tokens = [...bpe.encode(prompt.toUpperCase()), bpe.SEP];
  const out = [];
  const counts = new Map();
  for (let i = 0; i < MAX_NEW; i++) {
    const logits = forward(api, sec, arch, [...tokens, ...out], base);
    // repetition penalty
    for (const [tid] of counts) logits[tid] /= REP_PENALTY;
    if (TEMP <= 0) {
      let best = 0, bv = -Infinity;
      for (let j = 0; j < logits.length; j++) if (logits[j] > bv) { bv = logits[j]; best = j; }
      if (best === bpe.EOS) break;
      out.push(best); counts.set(best, (counts.get(best) || 0) + 1);
      continue;
    }
    // temperature softmax sampling
    let mx = -Infinity;
    for (let j = 0; j < logits.length; j++) if (logits[j] > mx) mx = logits[j];
    let sum = 0; const probs = new Float64Array(logits.length);
    for (let j = 0; j < logits.length; j++) { const e = Math.exp((logits[j] - mx) / TEMP); probs[j] = e; sum += e; }
    let r = mulberry32() * sum, pick = 0;
    for (let j = 0; j < probs.length; j++) { r -= probs[j]; if (r <= 0) { pick = j; break; } }
    if (pick === bpe.EOS) break;
    out.push(pick); counts.set(pick, (counts.get(pick) || 0) + 1);
  }
  return out.map(t => bpe.idToToken.get(t) ?? `[${t}]`).join('').trim();
}

const SET = process.env.EVAL_SET || 'eval_set.jsonl';
const lines = readFileSync(new URL('./' + SET, import.meta.url), 'utf8').trim().split('\n');
const results = [];
for (const line of lines) {
  const { id, category, prompt } = JSON.parse(line);
  const response = sample(prompt);
  results.push({ id, category, prompt, response });
  console.error(`  [${id}] ${response.slice(0, 60).replace(/\n/g, ' ')}`);
}
writeFileSync(new URL(`./out_${tag}.jsonl`, import.meta.url), results.map(r => JSON.stringify(r)).join('\n') + '\n');
console.error(`\nwrote eval/out_${tag}.jsonl (${results.length} responses, model=${prefix}, temp=${TEMP}, seed=${seedArg || 1234})`);
