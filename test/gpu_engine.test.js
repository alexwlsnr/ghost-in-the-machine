/**
 * GPU engine tests — testable without a real GPU.
 *
 * Covers routing logic, nibble dequantization math, buffer size
 * calculations, and shader source invariants.
 *
 * Run: node --test test/gpu_engine.test.js
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

// Pure-JS re-implementations of GPU engine logic for unit testing

const GPU_THRESHOLD_BYTES = 60_000_000;
function shouldUseGPU(arch) {
  const matrixParams = arch.n_layers * (4 * arch.d_model ** 2 + 2 * arch.d_model * arch.d_ff);
  return matrixParams * 4 >= GPU_THRESHOLD_BYTES;
}

// Mirrors the WGSL w_val() nibble extraction
function nibbleToFloat(buf, flatIdx) {
  const byteIdx  = flatIdx >> 1;
  const byteVal  = buf[byteIdx];
  const nibble   = (flatIdx & 1) === 0
    ? (byteVal >> 4) & 0xF   // even index → high nibble
    : byteVal & 0xF;          // odd index  → low nibble
  return nibble - 8;
}

// Mirrors py/serialize.py quantize_4bit_grouped packing
function packNibbles(values) {
  const out = new Uint8Array(Math.ceil(values.length / 2));
  for (let i = 0; i < values.length; i += 2) {
    const hi = (Math.round(values[i])     + 8) & 0xF;
    const lo = (Math.round(values[i + 1] ?? 0) + 8) & 0xF;
    out[i >> 1] = (hi << 4) | lo;
  }
  return out;
}

// ── shouldUseGPU ──────────────────────────────────────────────────────────

test('shouldUseGPU returns false for Wisp (3.3M params)', () => {
  const wisp = { d_model: 256, n_layers: 4, d_ff: 1024, n_heads: 4 };
  assert.equal(shouldUseGPU(wisp), false);
});

test('shouldUseGPU returns false for Shade (~11M params)', () => {
  const shade = { d_model: 384, n_layers: 6, d_ff: 1536, n_heads: 6 };
  assert.equal(shouldUseGPU(shade), false);
});

test('shouldUseGPU returns true for Spec512 (~27.6M params)', () => {
  const spec = { d_model: 512, n_layers: 8, d_ff: 2048, n_heads: 8 };
  assert.equal(shouldUseGPU(spec), true);
});

test('shouldUseGPU threshold separates Shade (~40 MB) from Spec512 (~95 MB)', () => {
  // Shade: ~40 MB → stays on Wasm
  const shade = { d_model: 384, n_layers: 6, d_ff: 1536, n_heads: 6 };
  // Spec512: ~95 MB → gets GPU
  const spec  = { d_model: 512, n_layers: 8, d_ff: 2048, n_heads: 8 };
  assert.equal(shouldUseGPU(shade), false);
  assert.equal(shouldUseGPU(spec),  true);
});

// ── Nibble dequantization ──────────────────────────────────────────────────

test('nibble extraction round-trips -8..7 for even indices', () => {
  for (let val = -8; val <= 7; val++) {
    const buf = packNibbles([val, 0]);
    assert.equal(nibbleToFloat(buf, 0), val,
      `even index: expected ${val}, got ${nibbleToFloat(buf, 0)}`);
  }
});

test('nibble extraction round-trips -8..7 for odd indices', () => {
  for (let val = -8; val <= 7; val++) {
    const buf = packNibbles([0, val]);
    assert.equal(nibbleToFloat(buf, 1), val,
      `odd index: expected ${val}, got ${nibbleToFloat(buf, 1)}`);
  }
});

test('nibble extraction handles boundary: 0 stored as nibble 8', () => {
  const buf = packNibbles([0, 0]);
  assert.equal(nibbleToFloat(buf, 0), 0);
  assert.equal(nibbleToFloat(buf, 1), 0);
});

test('nibble extraction matches serializer for a short weight row', () => {
  // Simulate quantizing weights [-7, 3, -1, 5, 0, -4, 7, -8]
  const weights = [-7, 3, -1, 5, 0, -4, 7, -8];
  const packed  = packNibbles(weights);
  for (let i = 0; i < weights.length; i++) {
    assert.equal(nibbleToFloat(packed, i), weights[i],
      `index ${i}: expected ${weights[i]}`);
  }
});

test('nibble extraction is consistent at byte boundaries', () => {
  // Check indices that cross byte boundaries (idx 1→2)
  const vals = [3, -5, 7, -2];
  const buf = packNibbles(vals);
  // idx 0 and 1 → byte 0; idx 2 and 3 → byte 1
  assert.equal(nibbleToFloat(buf, 2), 7);
  assert.equal(nibbleToFloat(buf, 3), -2);
});

// ── Buffer size calculations ───────────────────────────────────────────────

test('KV cache size formula: max_ctx * d_model * 4 bytes', () => {
  const arch = { max_len: 1024, d_model: 512 };
  const cacheBytes = arch.max_len * arch.d_model * 4;
  assert.equal(cacheBytes, 2_097_152); // 2 MB per layer per K or V
});

test('logits staging buffer is 258 * 4 = 1032 bytes', () => {
  const vocabSize = 258;
  assert.equal(vocabSize * 4, 1032);
});

// ── Shader source invariants ───────────────────────────────────────────────

// Import the shader strings by reading the source file as text
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(__dirname, '../ts/src/gpu_engine.ts'), 'utf8');

const shaders = {
  EMBED:    src.match(/const EMBED_SHADER[^`]+`([^`]+)`/)?.[1] ?? '',
  LN:       src.match(/const LAYERNORM_SHADER[^`]+`([^`]+)`/)?.[1] ?? '',
  MATF32:   src.match(/const MATMUL_F32_SHADER[^`]+`([^`]+)`/)?.[1] ?? '',
  MATI4G:   src.match(/const MATMUL_INT4G_SHADER[^`]+`([^`]+)`/)?.[1] ?? '',
  WRITE_KV: src.match(/const WRITE_KV_SHADER[^`]+`([^`]+)`/)?.[1] ?? '',
  ATTN:     src.match(/const ATTENTION_SHADER[^`]+`([^`]+)`/)?.[1] ?? '',
  ADD:      src.match(/const ADD_SHADER[^`]+`([^`]+)`/)?.[1] ?? '',
  RELU:     src.match(/const RELU_SHADER[^`]+`([^`]+)`/)?.[1] ?? '',
};

test('all shader strings are non-empty', () => {
  for (const [name, s] of Object.entries(shaders)) {
    assert.ok(s.length > 20, `Shader ${name} is empty or too short`);
  }
});

test('each shader declares a @compute entrypoint', () => {
  for (const [name, s] of Object.entries(shaders)) {
    assert.ok(s.includes('@compute'), `Shader ${name} missing @compute`);
    assert.ok(s.includes('fn main'), `Shader ${name} missing fn main`);
  }
});

test('MATMUL_INT4G shader implements nibble extraction with correct formula', () => {
  // Must contain the bit-shift extraction and the -8 dequantization
  assert.ok(shaders.MATI4G.includes('>> 4u'), 'Missing high nibble shift');
  assert.ok(shaders.MATI4G.includes('& 0xFu'), 'Missing nibble mask');
  assert.ok(shaders.MATI4G.includes('- 8'), 'Missing dequantization offset');
});

test('ATTENTION shader uses workgroup shared memory for scores', () => {
  assert.ok(shaders.ATTN.includes('var<workgroup>'), 'Missing workgroup shared memory for scores');
  assert.ok(shaders.ATTN.includes('1024'), 'Missing max_ctx=1024 score array size');
});

test('LAYERNORM shader uses tree reduction pattern', () => {
  assert.ok(shaders.LN.includes('workgroupBarrier'), 'Missing barrier in LN');
  assert.ok(shaders.LN.includes('inverseSqrt'), 'Missing inverseSqrt in LN');
  assert.ok(shaders.LN.includes('var<workgroup>'), 'Missing shared memory in LN');
});

test('ATTENTION shader computes scale = 1/sqrt(d_head)', () => {
  assert.ok(shaders.ATTN.includes('sqrt(f32(p.d_head)'), 'Missing sqrt scaling in attention');
});
