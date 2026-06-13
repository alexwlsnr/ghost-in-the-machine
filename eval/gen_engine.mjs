// Headless eval generator using the DEPLOYED inference engine.
//   node eval/gen_engine.mjs <model_prefix> <out_tag> [seed]
//   EVAL_SET=eval_set_200.jsonl node eval/gen_engine.mjs model_spectre_v2_ep60 ep60 1234
//
// Imports the SAME compiled engine + generate() the browser UI uses
// (dist/tier2_transformer.js), so output is bug-for-bug faithful to deployment
// and fast (KV-cached prefill + forwardIncremental, ~70 tps). Sampling params
// (temp/top_k/top_p/max_new) are read from dist/models.json by bin name, exactly
// as the UI does. A seeded RNG replaces Math.random for reproducibility.
//
// Node runs the WASM/SIMD path (no WebGPU) — faithful to WASM-path users and all
// shared logic; WebGPU-kernel-specific behavior is not exercised.
import { readFileSync, writeFileSync } from 'fs';
import { instantiateModel, generate, createCache } from '../dist/tier2_transformer.js';

Object.defineProperty(globalThis, 'navigator', { value: { gpu: null }, configurable: true, writable: true });
globalThis.performance = globalThis.performance ?? { now: () => Date.now() };

const [prefix, tag, seedArg] = process.argv.slice(2);
if (!prefix || !tag) { console.error('usage: node eval/gen_engine.mjs <model_prefix> <out_tag> [seed]'); process.exit(1); }
const SET = process.env.EVAL_SET || 'eval_set.jsonl';

let seed = (seedArg ? parseInt(seedArg) : 1234) >>> 0;
function rng() {
  seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

// Deployed sampling params, looked up by bin filename (as the UI does via its registry).
const mj = JSON.parse(readFileSync(new URL('../dist/models.json', import.meta.url), 'utf8'));
const list = Array.isArray(mj) ? mj : (mj.models || [].concat(...Object.values(mj)));
const reg = (Array.isArray(list) ? list : []).find(m => m && m.bin === prefix + '.bin') || {};
const temp = reg.temp ?? 0.8, topK = reg.top_k ?? 0, topP = reg.top_p ?? 1.0;
const maxNew = reg.max_new ?? (Number(reg.ctx) >= 512 ? 120 : 160);

const toAB = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const wasmBuf = readFileSync(new URL('../dist/tier2_kernel_simd.wasm', import.meta.url));
const model = await instantiateModel(
  wasmBuf,
  toAB(readFileSync(new URL(`../dist/${prefix}.bin`, import.meta.url))),
  toAB(readFileSync(new URL(`../dist/${prefix}.json`, import.meta.url))),
);

const lines = readFileSync(new URL('./' + SET, import.meta.url), 'utf8').trim().split('\n');
const out = [];
const t0 = Date.now();
for (const line of lines) {
  const e = JSON.parse(line);
  const history = e.history;            // [{q,r}, ...] for multi-turn, else undefined
  const cache = createCache(model);
  let resp = '';
  for await (const step of generate(model, e.prompt, maxNew, temp, rng, cache, topK, topP, history, undefined, true)) {
    if (step.done) break;
    resp += step.char;
  }
  // For the judge, show full conversation context on multi-turn items.
  const shown = history && history.length
    ? history.map(h => `User: ${h.q}\nAssistant: ${h.r}`).join('\n') + `\nUser: ${e.prompt}`
    : e.prompt;
  out.push({ id: e.id, category: e.category, prompt: shown, response: resp.trim() });
  console.error(`  [${e.id}] ${resp.slice(0, 60).replace(/\n/g, ' ')}`);
}
const dt = ((Date.now() - t0) / 1000).toFixed(1);
writeFileSync(new URL(`./out_${tag}.jsonl`, import.meta.url), out.map(o => JSON.stringify(o)).join('\n') + '\n');
console.error(`\nwrote eval/out_${tag}.jsonl (${out.length}) — model=${prefix} temp=${temp} topK=${topK} topP=${topP} maxNew=${maxNew} | ${dt}s`);
