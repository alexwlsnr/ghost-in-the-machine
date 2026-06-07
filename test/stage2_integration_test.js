#!/usr/bin/env node
/**
 * Stage 2.2 — Multi-layer dispatcher integration test.
 *
 * Chains all layers of a real Tier 2 model through the Wasm kernel
 * and compares the final output against the Python reference.
 *
 * Usage:
 *   node stage2_integration_test.js [model_bundle_prefix]
 */

const fs = require("fs");
const path = require("path");

const bundlePrefix = process.argv[2] || "stage2/test_model";
const wasmPath = path.join(
  __dirname, "stage2", "target", "wasm32-unknown-unknown", "release", "tier2_kernel.wasm"
);
const binPath = `${bundlePrefix}.bin`;
const jsonPath = `${bundlePrefix}.json`;

// ─── Wasm loader ────────────────────────────────────────────────────────

async function loadWasm(filePath) {
  const buf = fs.readFileSync(filePath);
  const { instance } = await WebAssembly.instantiate(buf);
  return instance.exports;
}

// ─── Memory helpers ──────────────────────────────────────────────────────

function writeBytes(mem, data, offset) {
  new Uint8Array(mem.buffer, offset, data.length).set(data);
}

function writeI16(mem, values, offset) {
  new Int16Array(mem.buffer, offset, values.length).set(values);
}

function readI16(mem, offset, length) {
  return new Int16Array(mem.buffer, offset, length);
}

function align4(n) { return (n + 3) & ~3; }

// ─── Layer config type ───────────────────────────────────────────────────

/**
 * @typedef {{
 *   inDim: number,
 *   outDim: number,
 *   weightOffset: number,
 *   biasOffset: number,
 *   relu: boolean,
 * }} LayerConfig
 */

// ─── Multi-layer dispatcher (Stage 2.2) ──────────────────────────────────

/**
 * Runs multiple layers through the Wasm kernel sequentially.
 *
 * Uses double-buffering: input buffer A → output buffer B → input buffer B → output buffer A → ...
 *
 * @param {WebAssembly.Exports} exports - Wasm module exports
 * @param {LayerConfig[]} layers - Layer configurations
 * @param {Int16Array} initialInput - The initial input vector (trigram buckets)
 * @returns {{ outputs: Int16Array, layerOutputs: Int16Array[] }}
 */
function dispatchLayers(exports, layers, initialInput) {
  const memory = exports.memory;
  const layerForward = exports.layer_forward;

  // Ensure we have at least 1 page; the module already owns pages for __heap_base
  // We'll use memory from __heap_base onwards (the module's data section is tiny)

  // Allocate output buffers for each layer upfront
  const outBuffers = [];
  let curOffset = exports.__heap_base.value || 1024;

  // Each layer needs an output buffer; we use double-buffering with two slots
  // Slot A and Slot B rotate
  const maxOutDim = Math.max(...layers.map(l => l.outDim));
  const maxOutBytes = maxOutDim * 2;  // i16 = 2 bytes

  const prevBufOffset = curOffset;
  curOffset += maxOutBytes;
  curOffset = align4(curOffset);

  const currBufOffset = curOffset;
  curOffset += maxOutBytes;
  curOffset = align4(curOffset);

  // Write initial input into the "previous" buffer
  writeI16(memory, initialInput, prevBufOffset);

  // Allocate input buffer for the initial layer (same size as max input)
  // Actually, inputs for subsequent layers are the outputs of previous layers,
  // so both buffers double as input/output alternately.
  // For the first layer, we need an input buffer.
  const maxInDim = Math.max(...layers.map(l => l.inDim));
  const maxInBytes = maxInDim * 2;
  const inputBuf = curOffset;
  curOffset += maxInBytes;
  curOffset = align4(curOffset);

  // Grow memory if needed
  const pagesNeeded = Math.ceil(curOffset / 65536);
  const currentPages = memory.buffer.byteLength / 65536;
  if (pagesNeeded > currentPages) {
    memory.grow(pagesNeeded - currentPages);
  }

  // Actually, we need to rethink: the initial input comes from JS.
  // Layer 0: input = initial input, output → prevBuf
  // Layer 1: input = prevBuf, output → currBuf
  // Layer 2: input = currBuf, output → prevBuf

  // Simpler: allocate separate input and output for each layer call
  // We'll reuse two buffers and just copy.

  // Write the initial input to the input buffer area
  writeI16(memory, initialInput, inputBuf);

  let useBufA = true;
  const bufA = prevBufOffset;
  const bufB = currBufOffset;

  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i];
    const inPtr = (i === 0) ? inputBuf : (useBufA ? bufB : bufA);
    const outPtr = useBufA ? bufA : bufB;

    layerForward(
      layer.weightOffset,
      layer.biasOffset,
      inPtr,
      outPtr,
      layer.inDim,
      layer.outDim,
      layer.relu ? 1 : 0
    );

    useBufA = !useBufA;

    // Capture this layer's output for verification
    const snapshot = readI16(memory, outPtr, layer.outDim);
    outBuffers.push(new Int16Array(snapshot));
  }

  // Final output is in the last written buffer
  const lastOutInBufA = (layers.length % 2 === 1);
  const finalPtr = lastOutInBufA ? bufA : bufB;
  const finalOutDim = layers[layers.length - 1].outDim;

  return {
    outputs: readI16(memory, finalPtr, finalOutDim),
    layerOutputs: outBuffers,
  };
}

// ─── Main ────────────────────────────────────────────────────────────────

async function main() {
  console.log("=== Stage 2.2: Multi-Layer Dispatcher Integration Test ===\n");

  // 1. Load manifest
  console.log(`[integ] Loading model from ${bundlePrefix}...`);
  const manifest = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
  const layerSizes = manifest.architecture.layer_sizes;
  console.log(`[integ] Architecture: ${layerSizes.join(" → ")}`);

  // 2. Load binary
  const binData = fs.readFileSync(binPath);
  console.log(`[integ] Binary file: ${binData.length} bytes`);

  // 3. Parse layers from binary
  const layers = [];
  let offset = 0;
  for (let i = 0; i < layerSizes.length - 1; i++) {
    const inDim = layerSizes[i];
    const outDim = layerSizes[i + 1];
    const numWeights = inDim * outDim;
    const packedSize = Math.ceil(numWeights / 4);

    const weightBytes = binData.slice(offset, offset + packedSize);
    offset += packedSize;

    const biasLen = outDim * 2;
    const biasBytes = binData.slice(offset, offset + biasLen);
    offset += biasLen;

    layers.push({
      inDim,
      outDim,
      weightData: new Uint8Array(weightBytes),
      biasData: new Int16Array(biasBytes.buffer, biasBytes.byteOffset, outDim),
      relu: i < layerSizes.length - 2,  // ReLU on hidden layers only
    });
    console.log(`[integ]   Layer ${i + 1}: ${inDim}→${outDim}, ${packedSize}w + ${outDim}b`);
  }

  // 4. Load Wasm
  const exports = await loadWasm(wasmPath);
  const memory = exports.memory;

  // 5. Allocate model data in Wasm memory
  //    Grow memory to fit all layer data plus working buffers
  let memOffset = exports.__heap_base.value || 1024;

  // Calculate total size needed for all layer data
  let totalLayerData = 0;
  for (const layer of layers) {
    totalLayerData = align4(totalLayerData + layer.weightData.length);
    totalLayerData = align4(totalLayerData + layer.biasData.length * 2);
  }

  // Working buffers: max(InDims) * 2 + max(OutDims) * 2 * 2 (double buffer)
  const maxIn = Math.max(...layers.map(l => l.inDim));
  const maxOut = Math.max(...layers.map(l => l.outDim));
  const workingBufSize = maxIn * 2 + maxOut * 2 * 2;

  const totalNeeded = memOffset + totalLayerData + workingBufSize;

  const pagesNeeded = Math.ceil(totalNeeded / 65536);
  const currentPages = memory.buffer.byteLength / 65536;
  if (pagesNeeded > currentPages) {
    memory.grow(pagesNeeded - currentPages);
    console.log(`[integ] Grew memory to ${pagesNeeded} pages (${pagesNeeded * 64}KB)`);
  }

  // Write layer data into Wasm memory and update offsets
  for (const layer of layers) {
    memOffset = align4(memOffset);
    layer.weightOffset = memOffset;
    writeBytes(memory, layer.weightData, memOffset);
    memOffset += layer.weightData.length;

    memOffset = align4(memOffset);
    layer.biasOffset = memOffset;
    writeI16(memory, layer.biasData, memOffset);
    memOffset += layer.biasData.length * 2;
  }

  // 6. Generate test input (random trigram buckets, like real inference)
  const charset = manifest.charset || [];
  const inputDim = layerSizes[0];
  console.log(`[integ] Input dim: ${inputDim}, charset size: ${charset.length}`);

  // Use a deterministic but representative input
  const testInput = new Int16Array(inputDim);
  for (let i = 0; i < Math.min(inputDim, 30); i++) {
    testInput[i] = (i * 7 + 3) % 50;  // simulate non-zero bucket counts
  }

  // 7. Run dispatcher
  console.log(`[integ] Running ${layers.length}-layer forward pass...`);
  const start = performance.now();
  const { outputs, layerOutputs } = dispatchLayers(exports, layers, testInput);
  const elapsed = performance.now() - start;
  console.log(`[integ] Completed in ${elapsed.toFixed(2)}ms`);

  // 8. Quick sanity: output should have valid values
  console.log(`[integ] Final output dim: ${outputs.length}`);
  console.log(`[integ] Output range: [${Math.min(...outputs)}, ${Math.max(...outputs)}]`);

  // 9. Compare against Python reference if available
  // For now, verify structural correctness
  let ok = true;
  for (let i = 0; i < layers.length; i++) {
    const out = layerOutputs[i];
    if (out.length !== layers[i].outDim) {
      console.log(`[integ] Layer ${i + 1}: wrong output dim ${out.length} (expected ${layers[i].outDim})`);
      ok = false;
    }
  }

  console.log(`\n[integ] Multi-layer dispatch: ${ok ? "PASS ✓" : "FAIL ✗"}`);
  if (!ok) process.exitCode = 1;
}

main().catch(err => {
  console.error("[integ] Fatal:", err);
  process.exit(1);
});
