# Tokenizer formats

This project supports **three** tokenization schemes across its model zoo. They are
*not* interchangeable for a given model: a model's weights are bound to the exact
tokenizer (vocabulary **and** segmentation rules) it was trained on. Feeding a model
a different segmentation produces garbage even if every token exists in the vocab.

The browser engine, the Python training pipeline, and the GGUF/llama.cpp path all
understand which scheme a model uses and apply the matching rules.

## The three schemes

| # | Name | Manifest `tokenizer.type` | Algorithm | Used by |
|---|------|---------------------------|-----------|---------|
| 1 | **raw-byte** | *(no `tokenizer` section)* | ids = raw bytes (0–255) + PAD/EOS. No BPE. | earliest tier-2.5 models |
| 2 | **char-BPE** | absent or `"char"` | merge-rank BPE over raw Unicode chars, **literal spaces, no pretokenizer**, global-greedy merges | all current deployed BPE models — Wisp, Shade, Spectre (v1/v2) |
| 3 | **bytelevel** | `"bytelevel"` | GPT-2 byte-level BPE: bytes→unicode map, GPT-2 pretokenizer regex split, then per-chunk merges | new llama.cpp/GGUF-compatible models (e.g. Wisp byte-level retrain) |

Special tokens are always `<PAD>`=0, `<EOS>`=1, `<SEP>`=2. The chat format is
`Q <SEP> R <EOS>`, and multi-turn sequences pack as `Q1 <SEP> R1 <SEP> Q2 <SEP> ...`.

### Why scheme 2 is *not* llama.cpp-compatible
char-BPE has **no pretokenizer** and merges greedily across the whole string with
literal spaces. llama.cpp's BPE (and every standard BPE) splits text into words via
a regex first, then byte-level-encodes and merges per chunk. There is no GGUF
metadata flag that makes llama.cpp reproduce the char-BPE rules, so char-BPE models
can only run in llama.cpp by feeding externally-produced token IDs — not via the
embedded tokenizer. This is the gap scheme 3 was created to close.

### Why scheme 3 *is* llama.cpp-compatible
bytelevel is a standard GPT-2 byte-level BPE. A model trained on it embeds a
faithful tokenizer into its GGUF and runs in **stock** `llama-cli`/ollama/LM Studio
with no external glue. Verified: llama.cpp tokenizes byte-for-byte identically to
our `ByteBPETokenizer`, and full generation works through the embedded tokenizer.

## Where each scheme lives in the code

### Training (Python)
- `py/train_bpe.py` — trains a **char-BPE** tokenizer (legacy custom format:
  `{vocab, id_to_token, merges, special}`).
- `py/train_bpe_bytelevel.py` — trains a **bytelevel** tokenizer via HuggingFace
  `tokenizers` (ByteLevel pre-tokenizer + GPT-2 regex). Output is a standard HF
  `tokenizer.json`.
- `py/bpe_tokenizer.py` — `BPETokenizer`, the char-BPE loader/encoder.
- `py/byte_bpe_tokenizer.py` — `ByteBPETokenizer`, the bytelevel loader/encoder
  (wraps HF `tokenizers`). `is_hf_tokenizer(path)` detects the format.
- `py/train_transformer.py` — auto-detects the tokenizer format at `--tokenizer`
  and picks `ByteBPETokenizer` (if HF `tokenizer.json`) or `BPETokenizer` (legacy).
  Omitting `--tokenizer` trains a raw-byte model.

### Serialization → browser manifest (Python)
- `py/serialize.py` → `_normalize_tokenizer()` flattens whichever tokenizer the
  checkpoint references into one manifest schema and stamps `type`:
  - HF `tokenizer.json` (`model`+`added_tokens`) → `type: "bytelevel"`, byte-mapped
    `vocab`/`merges`, `special` from added tokens.
  - legacy char-BPE → `type: "char"`.
  - raw-byte models have no `tokenizer` section at all.
  Resulting manifest section: `{type, vocab_size, special, vocab, id_to_token, merges}`.

### Browser inference (TypeScript → `dist/`)
- `ts/src/bpe_tokenizer.ts`:
  - `BPETokenizer` — scheme 2 (char).
  - `ByteLevelBPETokenizer` — scheme 3 (GPT-2 byte-map + regex + per-chunk merges,
    with byte-unmapping on decode).
  - `makeTokenizer(data)` — picks the class from `data.type` (`"bytelevel"` →
    byte-level; otherwise char).
- `ts/src/tier2_transformer.ts` — calls `makeTokenizer(manifest.tokenizer)` when a
  `tokenizer` section is present; falls back to the built-in byte `encode()` (raw-byte)
  when it is absent.
- **Build:** `cd ts && npx tsc` emits to `ts/dist/`; copy the changed `*.js`/`*.d.ts`/
  `*.map` into the served repo-root `dist/`. (Keep `ts/dist` and `dist` in sync — see
  the `dist/ts drift hazard` note: never hand-edit `dist/*.js` without porting the
  change back to `ts/src`, or the next `tsc` reverts it.)

### GGUF / llama.cpp (Python)
- `py/convert_to_gguf.py` — converts a `ternary_modern` checkpoint to GGUF (Path A:
  dequantize ternary → F16, `llama` arch). Embeds the tokenizer: a bytelevel HF
  tokenizer is written faithfully as a GPT-2 GGUF tokenizer (`tokenizer.ggml.pre =
  "gpt-2"`); a char-BPE tokenizer is embedded best-effort but will **not** match
  llama.cpp tokenization (see above).
- `py/verify_gguf.py` — logit-parity check (PyTorch vs GGUF) on identical token IDs.
- `py/gen_gguf.py` — end-to-end generation via llama.cpp (Python tokenizer in front).
- `chat_gguf.py` — interactive REPL over a GGUF via llama.cpp (uses the embedded
  tokenizer for bytelevel models; works in stock llama.cpp).

Key conversion facts (the bits that bite): our interleaved RoPE == llama.cpp `llama`
NORM rope (no Q/K permute); fold the `sqrt(d_model)` embedding scale into
`token_embd.weight` and emit `output.weight` untied/unscaled; **norm weights must be
F32** (llama.cpp rejects f32×f16 in the RMSNorm multiply).

## Verifying tokenizer parity
- Python char vs bytelevel references and JS reproduction: `tok_parity.mjs`
  (compares `makeTokenizer().encode()` against Python `BPETokenizer` /
  `ByteBPETokenizer`). Both schemes confirmed identical.
- llama.cpp vs our bytelevel tokenizer: tokenize the same strings via a GGUF's
  embedded tokenizer (`Llama(...).tokenize`) and `ByteBPETokenizer.encode` —
  confirmed byte-for-byte identical.

## Decision history
We added scheme 3 to make models portable to the broader ecosystem (llama.cpp,
ollama, GGUF). Scheme 2 models keep working unchanged in the browser; scheme 3 is
the path forward when a model should also run in stock llama.cpp. See
`memory/gguf-interop-findings.md` for the running record.
