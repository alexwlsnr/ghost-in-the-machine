# Multi-Model Plan

## Three tiers

| Name | d_model | d_ff | Layers | ctx | Params | Float32 | 4-bit |
|------|---------|------|--------|-----|--------|---------|-------|
| **Wisp** (micro) | 256 | 1024 | 4 | 64 | 3.3M | 13 MB | ~1.7 MB |
| **Shade** (small) | 384 | 1536 | 6 | 128 | 10.9M | 42 MB | ~5.5 MB |
| **Specter** (large) | 768 | 3072 | 8 | 256 | 57.3M | 219 MB | ~28 MB |

All byte-level, vocab=258 (bytes 0-255 + PAD + EOS), pre-norm, ReLU FFN, learned pos embed.

## Training strategy

### Teacher model: gemma4-e2b-distill

Tested llama3.2-3b vs gemma4-e2b-distill (50-pair sample, 16 workers):

| Metric | llama3.2-3b | gemma4-e2b-distill |
|--------|-------------|-------------------|
| Valid pairs | 46/50 (92%) | 44/50 (88%) |
| Time (50 pairs) | 1.5s | 4.7s |
| Per-call latency | ~0.23s | ~0.30s |
| Conversational quality | Reflexive, asks questions back | Natural, empathetic |
| Deflection rate | High ("WHAT'S ON YOUR MIND?") | Low ("GREAT!", "NO WORRIES!") |

**Verdict: gemma4-e2b-distill wins on quality.** Slightly slower per call but
the conversational naturalness is dramatically better. With system prompt
support, it follows behavior instructions reliably.

### System prompt architecture

The critical insight: using the OpenAI `system` role instead of embedding
instructions in user messages makes the model *obey* behavioral constraints.
Tested and validated prompt:

```
You are a friendly, helpful AI assistant having a natural conversation.
Keep responses short (1-2 sentences, under 100 characters).
Match the user's tone — casual for casual, brief for brief.
Never ask follow-up questions. Never repeat the user's words back.
Just respond directly and naturally, like a real chat.
```

Key behavioral rules that fix llama3.2-3b's reflexive question-flipping:
- **"Never ask follow-up questions"** — prevents "HOW ARE YOU?" → "I'M GOOD, HOW ARE YOU?" loops
- **"Never repeat the user's words back"** — prevents "I NEED HELP" → "WHAT DO YOU NEED HELP WITH?"

Response generation uses a clean user template:
```
Respond to: {query}
```

### Data expansion pipeline

Current: 2K pairs — fine for 3.3M params, way too little for 11M/57M.
Target: 5K+ pairs for Wisp retrain, 50K+ for Shade, 200K+ for Specter.

**Approach: Template-driven seed generation + gemma4-e2b-distill for responses**

Wisp is NOT used for prompt generation (tested — ~50% garbage at 3.3M/2K).
Instead, prompts come from deterministic templates, and gemma4-e2b-distill
provides all responses via the system-prompt pipeline above.

1. **Template seed bank** — hand-write ~200 prompt templates across 10 categories:
   conversation, Q&A, jokes, recommendations, facts, creative, how-to, opinions,
   goodbyes, meta. Example: "Tell me a joke about {topic}",
   "How do I {verb} a {noun}", "What is the capital of {country}"

2. **Template expansion** — gemma4-e2b-distill expands 200 → 2,000 templates.
   Prompt: "Here are 5 chatbot prompt templates: [examples]. Generate 20 more
   on different topics. Output one per line."

3. **Slot filling** — for each template, randomly sample from word lists
   (100 topics, 200 verbs, 150 countries, etc.) to produce 5K-10K unique seed
   prompts

4. **Diversity pass** — for each seed prompt, gemma4-e2b-distill generates
   5 natural variations: "Given the prompt '{prompt}', generate 5 different
   ways a real user might phrase this." → 25K-50K distinct prompts

5. **Response generation** — feed all prompts through gemma4-e2b-distill
   with the system prompt above. Collect prompt|response pairs.

All intelligence steps (2, 4, 5) run through gemma4-e2b-distill with
16-64 parallel workers. Total gemma4-e2b-distill calls:
- Step 2: ~1 call (batch template generation)
- Step 4: 10K calls (one per seed prompt, 5 outputs each)
- Step 5: 50K calls (one per final prompt)
- **Total: ~60K calls, ~5 hours at 16 workers (~0.3s/call)**

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
