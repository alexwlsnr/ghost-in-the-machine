/**
 * Tier 2.5 Ghost Transformer — Float32 Orchestrator (fixed lengths)
 */
import type { GPUEngine } from './gpu_engine.js';
import type { Tokenizer } from './bpe_tokenizer.js';
export interface SectionDef {
    offset: number;
    size: number;
    shape: number[];
    dtype: string;
    scale?: number;
    scales_offset?: number;
    scales_size?: number;
    group_size?: number;
}
export interface Arch {
    vocab_size: number;
    d_model: number;
    n_heads: number;
    n_layers: number;
    d_ff: number;
    max_len: number;
    arch?: string;
    use_rope?: boolean;
    use_swiglu?: boolean;
    use_rmsnorm?: boolean;
}
interface WasmApi {
    memory: WebAssembly.Memory;
    matmul_ternary(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
    matmul_ternary_simd?(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
    matmul_ternary_sparse(counts: number, pos: number, neg: number, scale: number, b: number, inp: number, out: number, outD: number): void;
    ternary_convert_to_sparse?(w: number, counts: number, pos: number, neg: number, inD: number, outD: number): void;
    matmul_f32w(w: number, b: number, inp: number, out: number, inD: number, outD: number): void;
    softmax_f32(p: number, n: number): void;
    softmax_causal_f32(p: number, s: number): void;
    layer_norm_f32(x: number, g: number, b: number, d: number, e: number): void;
    rms_norm_f32(x: number, g: number, d: number, e: number): void;
    add_vec_f32(a: number, b: number, n: number): void;
    mul_vec_f32(a: number, b: number, n: number): void;
    relu_f32(p: number, n: number): void;
    silu_f32(p: number, n: number): void;
    attention_f32(qkv: number, scores: number, attn: number, seq: number, d: number, nHeads: number): void;
    matmul_8bit(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
    matmul_4bit(w: number, scale: number, b: number, inp: number, out: number, inD: number, outD: number): void;
    matmul_4bit_grouped(w: number, scales: number, b: number, inp: number, out: number, inD: number, outD: number, groupSize: number): void;
    matmul_bf16(w: number, b: number, inp: number, out: number, inD: number, outD: number): void;
}
/** Detect Wasm SIMD128 support at runtime (tiny probe Wasm module). */
export declare function detectSIMD(): Promise<boolean>;
export declare function loadModel(urls: {
    wasm: string;
    bin: string;
    json: string;
}): Promise<{
    api: WasmApi;
    manifest: Arch;
    sec: Record<string, SectionDef>;
    base: number;
    bpe: Tokenizer | undefined;
}>;
export declare function instantiateModel(wasmBuf: BufferSource, binBuf: ArrayBuffer, jsonBuf: ArrayBuffer): Promise<{
    api: WasmApi;
    manifest: Arch;
    sec: Record<string, SectionDef>;
    base: number;
    bpe: Tokenizer | undefined;
}>;
export declare function encode(text: string): number[];
export declare function forward(api: WasmApi, sec: Record<string, SectionDef>, arch: Arch, tokens: number[], base: number): Float32Array;
export type LoadedModel = Awaited<ReturnType<typeof loadModel>>;
export interface KVCache {
    /** Number of positions already written into the cache. */
    length: number;
    /** Per-layer K buffer: [max_len * d_model] floats. Index: layer → position * d + dim */
    k: Float32Array[];
    /** Per-layer V buffer: [max_len * d_model] floats. */
    v: Float32Array[];
}
export declare function createCache(model: LoadedModel): KVCache;
/** One turn of conversation history: the human query and the model's response. */
export interface Turn {
    q: string;
    r: string;
}
/**
 * Build the token sequence for a new query, prepending conversation history.
 *
 * Layout with 1 history turn:
 *   [q_prev][SEP][r_prev][SEP][query][SEP]
 *
 * maxHistory caps how many turns are injected. Default 1 — Spec512 v1 was
 * trained on mostly 2-turn sequences so >1 history turn degrades quality.
 * Increase when a model is retrained with richer multi-turn data.
 *
 * Turns are added newest-first; oldest turns are silently dropped when the
 * context budget (maxLen) would be exceeded.
 */
export declare function buildContextTokens(history: Turn[], query: string, maxLen: number, maxHistory?: number, bpe?: Tokenizer): number[];
export interface Step {
    char: string;
    token: number;
    done: boolean;
}
/**
 * Sample a token index from logits with temperature + optional top-k / top-p.
 * topK=0 / topP=1.0 → pure temperature (default, backward-compatible).
 * Recommended for Shade/Specter: topK=40, topP=0.9.
 *
 * repPenalty > 1.0 penalises tokens that appeared in recentTokens by dividing
 * their logit by the penalty before softmax. 1.3 is subtle, 1.5 is aggressive.
 */
export declare function sampleFromLogits(logits: Float32Array, temp: number, topK: number, topP: number, rand: () => number, repPenalty?: number, recentTokens?: number[]): number;
export declare function generate(model: Awaited<ReturnType<typeof loadModel>>, prompt: string, maxNew?: number, temp?: number, rand?: () => number, cache?: KVCache, topK?: number, topP?: number, history?: Turn[], gpuEngine?: GPUEngine, punctStop?: boolean): AsyncGenerator<Step>;
export {};
//# sourceMappingURL=tier2_transformer.d.ts.map