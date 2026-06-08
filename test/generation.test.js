/**
 * Generation golden: drives the real public generate() with a seeded RNG so the
 * orchestrator loop (sampling, EOS/PAD stop, context-window stop) is reproducible.
 *
 * Locks the CURRENT model's decoded output for fixed prompts/seed. The rework must
 * keep these strings identical (the model + bundle don't change in pre-work).
 *
 * Run: node --test test/generation.test.js
 */

import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { instantiateModel, generate, createCache } from '../dist/tier2_transformer.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const BUILT = join(ROOT, 'wasm/target/wasm32-unknown-unknown/release/tier2_kernel.wasm');
const WASM = existsSync(BUILT) ? BUILT : join(ROOT, 'dist/tier2_kernel.wasm');

const toAB = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);

// Deterministic LCG in [0,1) — injected in place of Math.random.
function lcg(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; };
}

let model;
before(async () => {
  const wasm = readFileSync(WASM);
  const bin = toAB(readFileSync(join(ROOT, 'dist/transformer_model.bin')));
  const json = toAB(readFileSync(join(ROOT, 'dist/transformer_model.json')));
  model = await instantiateModel(wasm, bin, json);
});

async function run(prompt, seed, temp = 1.0) {
  let out = '';
  for await (const step of generate(model, prompt, 160, temp, lcg(seed))) {
    if (step.done) break;
    out += step.char;
  }
  return out;
}

async function runWithCache(prompt, seed, temp = 1.0, cache) {
  let out = '';
  for await (const step of generate(model, prompt, 160, temp, lcg(seed), cache)) {
    if (step.done) break;
    out += step.char;
  }
  return out;
}

test('generate() is reproducible given a seeded RNG', async () => {
  // temp=2.0 — high enough that the (peaked) model's sampling genuinely varies,
  // so this only passes if generate() actually consumes the injected RNG.
  const a = await run('TELL ME A JOKE', 12345, 2.0);
  const b = await run('TELL ME A JOKE', 12345, 2.0);
  assert.equal(a, b);
});

// Golden decoded outputs — Wisp v3 (SEP-trained) at seed 777, temp 1.0.
// Updated when: model retrained (v3, with Q/R separator), SEP injected in
// generate(), or reference bundle changed.
const GOLDEN = [
  ['HELLO',          "HEY WHATS UP? HOWS YOUR DAY GOING SO FAR?"],
  ['TELL ME A JOKE', "HEY THERE! HOW'S IT GOING?"],
  ['GOODBYE',        'TAKE CARE AND TALK TO YOU SOON!'],
  ['HOW ARE YOU',    'HELLO THERE! DOING GREAT TODAY.'],
];

test('generate() reproduces the golden decoded outputs', async () => {
  for (const [prompt, expected] of GOLDEN) {
    assert.equal(await run(prompt, 777, 1.0), expected, `prompt: ${prompt}`);
  }
});

test('KV cache produces identical output to full-recompute', async () => {
  for (const [prompt] of GOLDEN) {
    const withoutCache = await run(prompt, 777, 1.0);
    const cache = createCache(model);
    const withCache = await runWithCache(prompt, 777, 1.0, cache);
    assert.equal(withCache, withoutCache, `KV cache mismatch for prompt: ${prompt}`);
  }
});
