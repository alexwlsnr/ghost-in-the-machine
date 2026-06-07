/**
 * Arch-driven forward: the orchestrator must run on any architecture, not just
 * Wisp (256d/64ctx). Builds a synthetic zero-weight bundle for a large arch
 * (long context) entirely in JS — no torch, no committed fixture — and checks
 * forward() completes without an out-of-bounds memory access.
 *
 * This is the regression test for:
 *   - #4 scratch margin: memory must be sized from arch, not a hardcoded 8 MB
 *     (this large arch needs ~13 MB of scratch and traps under the old margin).
 *   - #2 arch-driven dims: vocab_size / max_len / d / d_ff all differ from Wisp.
 *
 * Run: node --test test/forward_arch.test.js
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { instantiateModel, forward } from '../dist/tier2_transformer.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const BUILT = join(ROOT, 'wasm/target/wasm32-unknown-unknown/release/tier2_kernel.wasm');
const WASM = existsSync(BUILT) ? BUILT : join(ROOT, 'dist/tier2_kernel.wasm');

// Build a manifest + zero-filled .bin matching the section layout forward() reads.
function synthBundle(arch) {
  const { d_model: d, d_ff: ff, vocab_size: v, n_layers: L, max_len: m } = arch;
  const sections = {};
  let off = 0;
  const add = (name, floats) => {
    sections[name] = { offset: off, size: floats * 4, shape: [floats], dtype: 'float32' };
    off += floats * 4;
  };
  add('token_embed', v * d);
  add('pos_embed', m * d);
  for (let li = 0; li < L; li++) {
    const p = `enc${li}`;
    add(`${p}_ln1_w`, d); add(`${p}_ln1_b`, d);
    for (const n of ['q', 'k', 'v', 'o']) { add(`${p}_${n}_weight`, d * d); add(`${p}_${n}_bias`, d); }
    add(`${p}_ff1_weight`, ff * d); add(`${p}_ff1_bias`, ff);
    add(`${p}_ff2_weight`, d * ff); add(`${p}_ff2_bias`, d);
    add(`${p}_ln2_w`, d); add(`${p}_ln2_b`, d);
  }
  add('lnf_w', d); add('lnf_b', d);
  add('head_weight', v * d);

  const bin = new ArrayBuffer(off); // zero-initialized weights
  const json = new TextEncoder().encode(JSON.stringify({ architecture: arch, sections })).buffer;
  return { bin, json };
}

test('forward runs on a large long-context arch without OOB', async () => {
  // ~13 MB of forward scratch — exceeds the old hardcoded 8 MB margin, so this
  // traps unless memory is sized from the architecture.
  const arch = { vocab_size: 258, d_model: 256, n_heads: 4, n_layers: 1, d_ff: 1024, max_len: 1024 };
  const { bin, json } = synthBundle(arch);
  const model = await instantiateModel(readFileSync(WASM), bin, json);

  const seq = arch.max_len - 1;
  const tokens = Array.from({ length: seq }, (_, i) => i % 256);
  const logits = forward(model.api, model.sec, model.manifest, tokens, model.base);

  assert.equal(logits.length, arch.vocab_size);
  for (let i = 0; i < logits.length; i++) {
    assert.ok(Number.isFinite(logits[i]), `logit ${i} not finite`);
  }
});
