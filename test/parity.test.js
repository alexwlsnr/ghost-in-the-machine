/**
 * End-to-end logits parity: PyTorch (source of truth) vs the TS/Wasm forward pass.
 *
 * Locks the CURRENT float32 pipeline (serializer output + orchestrator + kernel) to
 * within tolerance of PyTorch. The serializer/TS rework must keep this green — a
 * regression like the serialize_v3 `pos_embed × √d` bug, a dropped scale, or an
 * off-by-one index would blow far past the tolerance.
 *
 * Reference data: test/fixtures/parity_logits.json (regenerate with
 *   .venv/bin/python3 test/gen_parity_fixtures.py).
 *
 * Run: node --test test/parity.test.js
 */

import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { instantiateModel, forward } from '../dist/tier2_transformer.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const BUILT = join(ROOT, 'wasm/target/wasm32-unknown-unknown/release/tier2_kernel.wasm');
const WASM = existsSync(BUILT) ? BUILT : join(ROOT, 'dist/tier2_kernel.wasm');

// Observed PyTorch-vs-Wasm divergence is ~4e-5; 2e-3 leaves headroom for cross-platform
// f32 noise while still catching any structural serializer/orchestrator regression.
const ATOL = 2e-3;

const toAB = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);

let model;
let fixture;

before(async () => {
  const wasm = readFileSync(WASM);
  const bin = toAB(readFileSync(join(ROOT, 'dist/transformer_model.bin')));
  const json = toAB(readFileSync(join(ROOT, 'dist/transformer_model.json')));
  model = await instantiateModel(wasm, bin, json);
  fixture = JSON.parse(readFileSync(join(ROOT, 'test/fixtures/parity_logits.json'), 'utf8'));
});

test('manifest architecture matches the reference checkpoint', () => {
  for (const k of ['vocab_size', 'd_model', 'n_heads', 'n_layers', 'd_ff', 'max_len']) {
    assert.equal(model.manifest[k], fixture.architecture[k], `arch.${k}`);
  }
});

test('forward logits match PyTorch within tolerance, on every prompt', () => {
  for (const c of fixture.cases) {
    const got = forward(model.api, model.sec, model.manifest, c.tokens, model.base);
    assert.equal(got.length, c.logits_last.length, `${c.prompt}: vocab length`);

    let maxAbs = 0;
    let argmax = 0;
    let best = -Infinity;
    for (let i = 0; i < got.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(got[i] - c.logits_last[i]));
      if (got[i] > best) { best = got[i]; argmax = i; }
    }
    assert.ok(maxAbs <= ATOL, `${c.prompt}: max abs logit diff ${maxAbs} > ${ATOL}`);
    assert.equal(argmax, c.argmax_last, `${c.prompt}: greedy next-token argmax`);
  }
});
