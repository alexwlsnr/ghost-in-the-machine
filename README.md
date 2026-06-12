# Ghost in the Machine — Tier 2.5 µLM

A family of **tiny transformers** that run entirely in your browser via WebAssembly.
No server, no API keys, no GPU — models load in well under a second and stream
coherent conversational English locally.

**[Live demo →](https://alexwlsnr.github.io/ghost-in-the-machine/)**

> **⚠️ README staleness note.** The "Architecture / Training / Bundle" sections below
> describe the *original* 3.3M byte-level tier-2.5 model. The current zoo is much
> larger and uses BPE tokenizers and ternary weights — Wisp (~7.6M), Shade (~11M),
> Spectre v1/v2 (~35M), all `ternary_modern` (RoPE + RMSNorm + SwiGLU). Treat the
> sections below as historical until they're refreshed; the **Tokenizers** and
> **GGUF / llama.cpp interop** sections immediately below are current.

## Tokenizers

Three tokenization schemes are supported. A model's weights are bound to the exact
tokenizer it trained on, so the engine selects the right one per model via the
manifest's `tokenizer.type` field:

| Scheme | `tokenizer.type` | Notes |
|--------|------------------|-------|
| raw-byte | *(no `tokenizer` section)* | ids = bytes; earliest models |
| char-BPE | absent / `"char"` | legacy custom BPE; all current Wisp/Shade/Spectre models; **not** llama.cpp-compatible |
| bytelevel | `"bytelevel"` | GPT-2 byte-level BPE; **llama.cpp/GGUF-compatible** |

Full details — algorithms, why char-BPE can't run in stock llama.cpp, and where each
scheme lives across training / serialization / browser / GGUF — are in
**[`docs/tokenizers.md`](docs/tokenizers.md)**.

## GGUF / llama.cpp interop

`ternary_modern` checkpoints convert to GGUF and run in stock llama.cpp:

```bash
.venv/bin/python3 py/convert_to_gguf.py --checkpoint ckpt/MODEL.pt \
    --tokenizer data/TOKENIZER.json --out dist/gguf/MODEL.gguf
.venv/bin/python3 py/verify_gguf.py --checkpoint ckpt/MODEL.pt --gguf dist/gguf/MODEL.gguf
.venv/bin/python3 chat_gguf.py dist/gguf/MODEL.gguf   # interactive REPL
```

Weights convert with 100% logit fidelity. For the *embedded* tokenizer to also be
faithful in stock `llama-cli`/ollama, the model must be trained on a **bytelevel**
tokenizer (char-BPE models run in llama.cpp only when fed external token IDs). See
[`docs/tokenizers.md`](docs/tokenizers.md).

## Architecture

| Layer | Detail |
|-------|--------|
| Type | Decoder-only byte-level transformer |
| Params | 3,307,520 |
| Config | d_model=256, 4 layers, 4 heads, d_ff=1024 |
| Vocab | 257 (all bytes + PAD), max context 64 tokens |
| Framework | PyTorch (training) → Rust/Wasm (inference) |
| Inference | ~36ms/token in JS/Wasm (no SIMD, no GPU) |

## Training

Self-distilled from **Llama 3.2 3B** (local llama-swap) on ~2K conversational prompt/response pairs. Trained for ~800 epochs with AdamW + cosine LR on an RTX 5080. Byte-level accuracy: ~93.8%.

```
HELLO            → HEY WHATS UP? HOWS YOUR DAY GOING SO FAR?
TELL ME A JOKE   → WHY DID THE SCARECROW WIN AN AWARD? BECAUSE HE WA...
RECOMMEND A MOVIE → PARASITE IS A THOUGHTPROVOKING AND HILARIOUS F...
GOODBYE          → HAVE A GREAT NIGHT AND SWEET DREAMS! BYE FOR NOW!
```

## Bundle

| File | Size | Purpose |
|------|------|---------|
| `tier2_kernel.wasm` | 4.5 KB | Wasm ops: matmul, softmax, layer norm, ReLU |
| `tier2_transformer.js` | 7.3 KB | TS orchestrator: forward, generate, sampling |
| `transformer_model.bin` | 13 MB | float32 weights (4-bit path targets ~1.7 MB) |
| `index.html` | 6.8 KB | CRT terminal demo |

Total page weight: **~13 MB** (loads in ~60ms on localhost, ~1s on fast CDN).

## Stack

- **Training:** PyTorch, AdamW, self-distillation from local Llama 3.2 3B
- **Inference kernel:** Rust → `wasm32-unknown-unknown` (no_std, `--release`, no allocator)
- **Orchestrator:** TypeScript, compiled to ES modules
- **Demo:** Vanilla HTML/CSS, CRT green-screen aesthetic

## Run locally

```bash
python3 -m http.server 8082 --directory dist
# Open http://localhost:8082
```

Or open `dist/index.html` directly in a browser (file:// works for most browsers).

## Repo structure

```
├── dist/                        # Deployable artifacts
│   ├── index.html               # CRT terminal demo
│   ├── tier2_transformer.js     # TS orchestrator (compiled)
│   ├── tier2_kernel.wasm        # Wasm kernel (compiled)
│   ├── transformer_model.bin    # float32 weights (13 MB)
│   └── transformer_model.json   # section manifest
├── wasm/                        # Rust Wasm kernel
│   ├── Cargo.toml
│   └── src/lib.rs               # matmul, softmax, layer_norm, relu
├── ts/                          # TypeScript orchestrator
│   ├── src/tier2_transformer.ts
│   └── tsconfig.transformer.json
├── py/                          # Training & distillation
│   ├── train_transformer.py     # Model definition + training loop
│   ├── distill.py               # Llama → training data self-distillation
│   ├── serialize_v3.py          # Model → .bin/.json serialization
│   └── training-data-transformer.txt
├── docs/                        # Planning & spec
│   ├── tier2_spec.md
│   └── tier2_PLAN.md
├── test/                        # Integration tests
└── .github/workflows/deploy.yml # CI → GitHub Pages
```

## Roadmap

- [ ] **4-bit quantization** — retrain with aligned QAT to hit ~1.7 MB bundle
- [ ] **KV caching** — avoid recomputing full sequence per token (5–10× faster)
- [ ] **8-bit Wasm SIMD** — accelerate matmul in the kernel
- [ ] **Multi-turn memory** — sliding window context beyond 64 tokens
