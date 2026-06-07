/**
 * Tier 2.5 Ghost Transformer -- Mixed-Precision Orchestrator
 *
 * Dispatches matmul calls based on per-section dtype from the manifest:
 *   dtype == "float32" -> matmul_f32w  (fp32 weights, no scale)
 *   dtype == "int8"    -> matmul_8bit  (i8 weights + per-tensor scale)
 *   dtype == "int4"    -> matmul_4bit  (4-bit packed weights + per-tensor scale)
 */
const PAD = 256;
const EOS = 257;
async function fetchBuf(url) {
    const r = await fetch(url);
    if (!r.ok)
        throw new Error(`HTTP ${r.status}: ${url}`);
    return r.arrayBuffer();
}
function forwardScratchBytes(arch) {
    const d = arch.d_model, ff = arch.d_ff, v = arch.vocab_size, seq = arch.max_len;
    const al = (n) => (n * 4 + 15) & ~15;
    return al(seq * d) + al(seq * d * 3) + al(seq * Math.max(d, ff))
        + al(d) + al(seq * seq) + al(seq * d) + al(v * 2);
}
export async function loadModel(urls) {
    const [wasmBuf, binBuf, jsonBuf] = await Promise.all([
        fetchBuf(urls.wasm), fetchBuf(urls.bin), fetchBuf(urls.json),
    ]);
    return instantiateModel(wasmBuf, binBuf, jsonBuf);
}
export async function instantiateModel(wasmBuf, binBuf, jsonBuf) {
    const wasm = await WebAssembly.instantiate(wasmBuf);
    const api = wasm.instance.exports;
    const manifest = JSON.parse(new TextDecoder().decode(jsonBuf));
    const sec = manifest.sections;
    const mem = api.memory;
    const heapBase = (wasm.instance.exports.__heap_base?.value ?? 0);
    const base = (heapBase + 15) & ~15;
    let maxOff = 0;
    for (const s of Object.values(sec))
        maxOff = Math.max(maxOff, s.offset + s.size);
    const margin = forwardScratchBytes(manifest.architecture) + 65536;
    const needPages = Math.ceil((base + maxOff + margin) / 65536);
    const curPages = mem.buffer.byteLength / 65536;
    if (needPages > curPages)
        mem.grow(needPages - curPages);
    const mem8 = new Uint8Array(mem.buffer);
    const bin8 = new Uint8Array(binBuf);
    for (const s of Object.values(sec)) {
        mem8.set(bin8.subarray(s.offset, s.offset + s.size), base + s.offset);
    }
    return { api, manifest: manifest.architecture, sec, base };
}
export function encode(text) {
    const t = [];
    for (let i = 0; i < text.length; i++) {
        const c = text.charCodeAt(i);
        if (c < 256)
            t.push(c);
    }
    return t;
}
export function forward(api, sec, arch, tokens, base) {
    const d = arch.d_model, nh = arch.n_heads, dh = d / nh, nl = arch.n_layers, seq = tokens.length, mem = api.memory;
    const S = (name) => base + sec[name].offset;
    const matmulWeights = (weightName, biasPtr, inp, out, inD, outD) => {
        const s = sec[weightName];
        const wPtr = S(weightName);
        if (s.dtype === 'int8') {
            api.matmul_8bit(wPtr, s.scale ?? 1.0, biasPtr, inp, out, inD, outD);
        }
        else if (s.dtype === 'int4') {
            api.matmul_4bit(wPtr, s.scale ?? 1.0, biasPtr, inp, out, inD, outD);
        }
        else {
            api.matmul_f32w(wPtr, biasPtr, inp, out, inD, outD);
        }
    };
    let off = base;
    for (const s of Object.values(sec))
        off = Math.max(off, base + s.offset + s.size);
    off = (off + 15) & ~15;
    const ba = (n) => { const o = off; off = (off + n * 4 + 15) & ~15; return o; };
    const f32 = (o, n) => new Float32Array(mem.buffer, o, n);
    const eOff = ba(seq * d);
    const qOff = ba(seq * d * 3);
    const tOff = ba(seq * Math.max(d, arch.d_ff));
    const lOff = ba(d);
    const sOff = ba(seq * seq);
    const aOff = ba(seq * d);
    const oOff = ba(arch.vocab_size * 2);
    const teW = f32(S('token_embed'), arch.vocab_size * d);
    const peW = f32(S('pos_embed'), arch.max_len * d);
    const emb = f32(eOff, seq * d);
    for (let p = 0; p < seq; p++) {
        const tid = tokens[p];
        for (let j = 0; j < d; j++)
            emb[p * d + j] = teW[tid * d + j] + peW[p * d + j];
    }
    for (let li = 0; li < nl; li++) {
        const pfx = `enc${li}`;
        for (let p = 0; p < seq; p++) {
            const lnBuf = f32(lOff, d);
            for (let j = 0; j < d; j++)
                lnBuf[j] = emb[p * d + j];
            api.layer_norm_f32(lOff, S(`${pfx}_ln1_w`), S(`${pfx}_ln1_b`), d, 1e-5);
            const qp = qOff + p * d * 3 * 4;
            matmulWeights(`${pfx}_q_weight`, S(`${pfx}_q_bias`), lOff, qp, d, d);
            matmulWeights(`${pfx}_k_weight`, S(`${pfx}_k_bias`), lOff, qp + d * 4, d, d);
            matmulWeights(`${pfx}_v_weight`, S(`${pfx}_v_bias`), lOff, qp + d * 8, d, d);
        }
        const qkv = f32(qOff, seq * d * 3);
        const attn = f32(aOff, seq * d);
        const scores = f32(sOff, seq * seq);
        attn.fill(0);
        for (let h = 0; h < nh; h++) {
            const ho = h * dh;
            for (let qi = 0; qi < seq; qi++) {
                for (let kj = 0; kj <= qi; kj++) {
                    let dot = 0;
                    for (let x = 0; x < dh; x++)
                        dot += qkv[qi * d * 3 + ho + x] * qkv[kj * d * 3 + d + ho + x];
                    scores[qi * seq + kj] = dot / Math.sqrt(dh);
                }
                for (let kj = qi + 1; kj < seq; kj++)
                    scores[qi * seq + kj] = -Infinity;
            }
            api.softmax_causal_f32(sOff, seq);
            for (let qi = 0; qi < seq; qi++) {
                for (let x = 0; x < dh; x++) {
                    let val = 0;
                    for (let kj = 0; kj < seq; kj++)
                        val += scores[qi * seq + kj] * qkv[kj * d * 3 + d * 2 + ho + x];
                    attn[qi * d + ho + x] = val;
                }
            }
        }
        for (let p = 0; p < seq; p++) {
            matmulWeights(`${pfx}_o_weight`, S(`${pfx}_o_bias`), aOff + p * d * 4, tOff + p * d * 4, d, d);
        }
        api.add_vec_f32(eOff, tOff, seq * d);
        for (let p = 0; p < seq; p++) {
            const lnBuf = f32(lOff, d);
            for (let j = 0; j < d; j++)
                lnBuf[j] = emb[p * d + j];
            api.layer_norm_f32(lOff, S(`${pfx}_ln2_w`), S(`${pfx}_ln2_b`), d, 1e-5);
            const up = tOff + p * arch.d_ff * 4;
            matmulWeights(`${pfx}_ff1_weight`, S(`${pfx}_ff1_bias`), lOff, up, d, arch.d_ff);
            api.relu_f32(up, arch.d_ff);
            matmulWeights(`${pfx}_ff2_weight`, S(`${pfx}_ff2_bias`), up, lOff, arch.d_ff, d);
            for (let j = 0; j < d; j++)
                emb[p * d + j] += f32(lOff, d)[j];
        }
    }
    const lp = seq - 1;
    const lnBuf = f32(lOff, d);
    for (let j = 0; j < d; j++)
        lnBuf[j] = emb[lp * d + j];
    api.layer_norm_f32(lOff, S('lnf_w'), S('lnf_b'), d, 1e-5);
    const zb = f32(oOff, arch.vocab_size);
    zb.fill(0);
    const lgOff = oOff + arch.vocab_size * 4;
    api.matmul_f32w(S('head_weight'), oOff, lOff, lgOff, d, arch.vocab_size);
    return f32(lgOff, arch.vocab_size);
}
export async function* generate(model, prompt, maxNew = 160, temp = 0.8, rand = Math.random) {
    const { api, manifest: arch, sec, base } = model;
    const win = arch.max_len - 1;
    const tokens = encode(prompt.toUpperCase()).slice(0, win);
    for (let s = 0; s < maxNew; s++) {
        if (tokens.length >= win) {
            yield { char: '', token: PAD, done: true };
            return;
        }
        await new Promise((r) => setTimeout(r, 0));
        const logits = forward(api, sec, arch, tokens, base);
        let maxV = -Infinity;
        for (let i = 0; i < arch.vocab_size; i++)
            if (logits[i] > maxV)
                maxV = logits[i];
        let sum = 0;
        const probs = new Float64Array(arch.vocab_size);
        for (let i = 0; i < arch.vocab_size; i++) {
            probs[i] = Math.exp((logits[i] - maxV) / temp);
            sum += probs[i];
        }
        let r = rand() * sum, next = 0;
        for (let i = 0; i < arch.vocab_size; i++) {
            r -= probs[i];
            if (r <= 0) {
                next = i;
                break;
            }
        }
        if (next === EOS || next === PAD) {
            yield { char: '', token: next, done: true };
            return;
        }
        tokens.push(next);
        yield { char: next < 256 ? String.fromCharCode(next) : '', token: next, done: false };
    }
    yield { char: '', token: PAD, done: true };
}
