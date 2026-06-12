// Tokenizers loaded from the model manifest's `tokenizer` section.
//
// Three tokenization schemes are supported across the model zoo:
//   1. raw-byte    — no `tokenizer` section in the manifest; the engine falls
//                    back to byte-level encode() in tier2_transformer.ts (ids = bytes).
//   2. char-BPE    — `{vocab, id_to_token, merges}` with no `type` (or type:"char").
//                    Legacy custom BPE: merges operate on raw Unicode characters,
//                    spaces are literal, no pretokenizer. Used by all current
//                    deployed BPE models (Wisp/Shade/Spectre).
//   3. bytelevel   — type:"bytelevel". GPT-2-style byte-level BPE: input bytes are
//                    mapped to printable unicode, split by the GPT-2 pretokenizer
//                    regex, then merged per-chunk. This is the llama.cpp/GGUF-
//                    compatible scheme — a model trained on it round-trips into
//                    stock llama.cpp and back.
//
// makeTokenizer() picks the right class from the manifest data.
/** Normalize merges to [a, b] pairs (HF byte-level stores them as "a b" strings). */
function normalizeMerges(merges) {
    return merges.map((m) => typeof m === 'string' ? m.split(' ') : m);
}
/**
 * Greedy merge-rank BPE: repeatedly merge the lowest-rank adjacent pair until
 * none remain. Shared by both BPE schemes (char operates on Unicode chars,
 * bytelevel on byte-mapped chars within a pretokenized chunk).
 */
function applyMerges(pieces, mergeRank, nMerges) {
    while (pieces.length >= 2) {
        let bestRank = nMerges;
        let bestI = -1;
        for (let i = 0; i < pieces.length - 1; i++) {
            const rank = mergeRank.get(pieces[i] + '\x00' + pieces[i + 1]) ?? nMerges;
            if (rank < bestRank) {
                bestRank = rank;
                bestI = i;
            }
        }
        if (bestI === -1)
            break;
        pieces[bestI] = pieces[bestI] + pieces[bestI + 1];
        pieces.splice(bestI + 1, 1);
    }
    return pieces;
}
// ─── Scheme 2: legacy char-level BPE ────────────────────────────────────────
// Merges over raw Unicode chars, literal spaces, no pretokenizer. Encode is
// O(text_len × n_merges) — fine for short prompts. Decode is O(1) per token.
export class BPETokenizer {
    constructor(data) {
        this.PAD = 0;
        this.EOS = 1;
        this.SEP = 2;
        this.vocab = new Map(Object.entries(data.vocab));
        this.idToToken = new Map(Object.entries(data.id_to_token).map(([k, v]) => [parseInt(k), v]));
        this.merges = normalizeMerges(data.merges);
        this.mergeRank = new Map(this.merges.map(([a, b], i) => [a + '\x00' + b, i]));
    }
    encode(text) {
        const pieces = applyMerges([...text], this.mergeRank, this.merges.length);
        return pieces.map((p) => this.vocab.get(p) ?? this.PAD);
    }
    decodeToken(id) {
        if (id <= 2)
            return '';
        return this.idToToken.get(id) ?? '';
    }
}
// ─── Scheme 3: GPT-2 byte-level BPE (llama.cpp / GGUF-compatible) ────────────
// GPT-2 byte↔unicode mapping: every byte 0-255 maps to a printable unicode char
// so the BPE alphabet is text-safe. Identical table to OpenAI's bytes_to_unicode.
function buildByteMaps() {
    const bs = [];
    for (let i = 33; i <= 126; i++)
        bs.push(i); // '!'..'~'
    for (let i = 161; i <= 172; i++)
        bs.push(i); // '¡'..'¬'
    for (let i = 174; i <= 255; i++)
        bs.push(i); // '®'..'ÿ'
    const cs = [...bs];
    let n = 0;
    for (let b = 0; b < 256; b++) {
        if (!bs.includes(b)) {
            bs.push(b);
            cs.push(256 + n);
            n++;
        }
    }
    const enc = new Map();
    const dec = new Map();
    for (let i = 0; i < bs.length; i++) {
        const ch = String.fromCodePoint(cs[i]);
        enc.set(bs[i], ch);
        dec.set(ch, bs[i]);
    }
    return { enc, dec };
}
// GPT-2 pretokenizer regex (contractions, letters, numbers, punctuation, spaces).
// Requires the /u flag for \p{L}/\p{N}.
const GPT2_PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
export class ByteLevelBPETokenizer {
    constructor(data) {
        this.enc = new TextEncoder();
        this.vocab = new Map(Object.entries(data.vocab));
        this.idToToken = new Map(Object.entries(data.id_to_token).map(([k, v]) => [parseInt(k), v]));
        this.merges = normalizeMerges(data.merges);
        this.mergeRank = new Map(this.merges.map(([a, b], i) => [a + '\x00' + b, i]));
        const maps = buildByteMaps();
        this.byteEnc = maps.enc;
        this.byteDec = maps.dec;
        const sp = data.special ?? {};
        this.PAD = sp.pad ?? this.vocab.get('<PAD>') ?? 0;
        this.EOS = sp.eos ?? this.vocab.get('<EOS>') ?? 1;
        this.SEP = sp.sep ?? this.vocab.get('<SEP>') ?? 2;
    }
    encode(text) {
        const ids = [];
        for (const match of text.matchAll(GPT2_PAT)) {
            // map this chunk's UTF-8 bytes through the byte↔unicode table
            let mapped = '';
            for (const b of this.enc.encode(match[0]))
                mapped += this.byteEnc.get(b);
            const pieces = applyMerges([...mapped], this.mergeRank, this.merges.length);
            for (const p of pieces)
                ids.push(this.vocab.get(p) ?? this.PAD);
        }
        return ids;
    }
    decodeToken(id) {
        if (id === this.PAD || id === this.EOS || id === this.SEP)
            return '';
        const tok = this.idToToken.get(id);
        if (tok === undefined)
            return '';
        // reverse the byte mapping, then UTF-8 decode (safe per-token for our
        // uppercase-ASCII corpus where each mapped char is a single byte)
        const bytes = new Uint8Array([...tok].map((ch) => this.byteDec.get(ch) ?? 0));
        return new TextDecoder().decode(bytes);
    }
}
/** Pick the tokenizer implementation from the manifest's tokenizer section. */
export function makeTokenizer(data) {
    return data.type === 'bytelevel'
        ? new ByteLevelBPETokenizer(data)
        : new BPETokenizer(data);
}
//# sourceMappingURL=bpe_tokenizer.js.map