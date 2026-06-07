# Multi-Model Plan

## Three tiers

| Name | d_model | d_ff | Layers | ctx | Params | Float32 | 4-bit |
|------|---------|------|--------|-----|--------|---------|-------|
| **Wisp** (micro) | 256 | 1024 | 4 | 64 | 3.3M | 13 MB | ~1.7 MB |
| **Shade** (small) | 384 | 1536 | 6 | 128 | 10.9M | 42 MB | ~5.5 MB |
| **Specter** (large) | 768 | 3072 | 8 | 256 | 57.3M | 219 MB | ~28 MB |

All byte-level, vocab=258 (bytes 0-255 + PAD + EOS), pre-norm, ReLU FFN, learned pos embed.

## Training strategy

### Data expansion needed
Current: 2K pairs — fine for 3.3M params, way too little for 11M/57M.
Target: 5K+ pairs for Wisp retrain, 50K+ for Shade, 200K+ for Specter.

**Approach: Template-driven + Llama completion**

Wisp is NOT reliable enough for prompt generation (tested — ~50% garbage at 3.3M/2K).
Instead, use a deterministic two-stage pipeline:

1. **Template seed bank** — hand-write ~200 prompt templates covering categories:
   conversation, Q&A, jokes, recommendations, facts, creative, code, math, how-to
   Example: "Tell me a joke about {topic}", "How do I {verb} a {noun}",
   "What is the capital of {country}", "Recommend a {category} for {occasion}"

2. **Template expansion** — Llama expands the 200 templates to 2,000 by:
   "Here are 5 prompt templates for a chatbot: [examples]. Generate 20 more
   on different topics. Be creative. Output one per line."

3. **Slot filling** — for each template, randomly sample from word lists
   (topics, verbs, countries, categories, etc.) to produce 5K-10K seed prompts

4. **Diversity pass** — feed seed prompts to Llama with: "Given this prompt
   template: '{template}', generate 5 different variations that a real user
   might ask." → 25K-50K prompts from 5K seeds

5. **Response generation** — feed all prompts to Llama 3.2 3B:
   "You are a helpful AI assistant. Respond naturally to: {prompt}"
   Collect prompt|response pairs

This is entirely Llama-driven after step 1-3 (deterministic template filling).
No tiny-model prompt generation. Total Llama calls:
- Step 2: ~1 call (batch)
- Step 4: 5K calls (one per seed prompt)  
- Step 5: 50K calls (one per final prompt)

At ~1s per Llama call, ~55K calls = ~15 hours. Feasible overnight.

### Training configs

**Wisp (existing, retrain if needed):**
- 2K-5K pairs, 800 epochs, lr=0.003, cosine schedule

**Shade (new):**
- 50K pairs, 200-400 epochs, lr=0.001, cosine schedule, warmup=10
- Batch size 32, gradient accumulation 2
- ~2-4 hours on RTX 5080

**Specter (new):**
- 200K pairs, 100-200 epochs, lr=0.0005, cosine schedule, warmup=20
- Batch size 16, gradient accumulation 4
- ~8-16 hours on RTX 5080

### Longer context training
Shade (ctx=128) and Specter (ctx=256) need training data that fills the context window:
- Generate prompt+response pairs that are up to ctx/2 tokens each
- During training, sequences are padded to the full context length
- Loss only computed on non-PAD tokens (EOS is not masked)

## Implementation checklist

### Phase 1: Infrastructure
- [ ] `train_transformer.py` already supports configurable arch (--d-model, --n-layers, etc.)
- [ ] Serialization script handles any arch (via manifest)
- [ ] Wasm kernel: no changes needed (matmul args are dynamic)
- [ ] TS orchestrator: arch-driven, no hardcoded dimensions
- [ ] Dist: `.bin`/`.json` per model (`model_wisp.bin`, `model_shade.bin`, `model_specter.bin`)

### Phase 2: Data expansion
- [ ] `expand_data.py` — template seed bank + Llama expansion pipeline
- [ ] Hand-write 200 prompt templates across 10 categories
- [ ] Template expansion via Llama (200 → 2,000 templates)
- [ ] Slot-filling with word lists (2K templates × 5 slots = 10K seed prompts)
- [ ] Diversity pass: Llama generates 5 variations per seed → 50K prompts
- [ ] Response generation: Llama answers all 50K prompts
- [ ] Filter: deduplicate, remove non-ASCII, enforce length bounds
- [ ] Validate: quick Wisp retrain on expanded data to check quality

### Phase 3: Training
- [ ] Train Wisp on expanded data (baseline)
- [ ] Train Shade (50K pairs, ~2-4h)
- [ ] Train Specter (200K pairs, ~8-16h)
- [ ] Evaluate: manual quality check on 10 test prompts each

### Phase 4: Serialization
- [ ] Serialize all 3 models to `.bin`/`.json`
- [ ] Validate bit-exact with Python forward pass

### Phase 5: UI
- [ ] Model switcher in terminal (tabs or dropdown)
- [ ] Default: Wisp (loads immediately)
- [ ] Shade/Specter load on demand (click to download & load)
- [ ] Show model name + size in status bar
- [ ] Loading progress for larger models
- [ ] Cache loaded models in memory (don't re-download on switch)

### Phase 6: Polish
- [ ] Update boot sequence to show available models
- [ ] Graceful fallback if model fails to load
- [ ] CI: build all 3 models, deploy to gh-pages
