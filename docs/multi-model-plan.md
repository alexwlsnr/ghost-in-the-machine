# Multi-Model Plan

## Three tiers

| Name | d_model | d_ff | Layers | heads | ctx | Params | Float32 | 4-bit |
|------|---------|------|--------|-------|-----|--------|---------|-------|
| **Wisp** (micro) | 256 | 1024 | 4 | 4 | 64 | 3.3M | 13 MB | ~1.7 MB |
| **Shade** (small) | 384 | 1536 | 6 | 6 | 128 | 10.9M | 42 MB | ~5.5 MB |
| **Specter** (large) | 768 | 3072 | 8 | 12 | 256 | 57.3M | 219 MB | ~28 MB |

All byte-level, vocab=258 (bytes 0-255 + PAD + EOS), pre-norm, ReLU FFN, learned pos embed.
Head count keeps `d_head = 64` at every tier (the TS forward divides `d_model / n_heads`
with no divisibility guard, so this must stay exact).

### Shipping constraints (read before committing to the table)

- **Specter at float32 (219 MB) cannot ship** — it exceeds GitHub's 100 MB per-file
  hard limit (can't even be committed), so Specter must be quantized to exist. Two
  shippable options, decided by eval (see *Quantization & precision selection*):
  **8-bit ~57 MB** (near-lossless, heavy download) or **4-bit ~28 MB** (lighter, some
  loss). A working quantization path is therefore a prerequisite for Specter, not a
  roadmap nicety.
- Wisp ships float32 today (13 MB). Shade float32 (42 MB) is shippable but heavy on a
  first paint; treat its 4-bit build as the real deliverable.
- **Open decision — is Specter worth d=768?** 200K short pairs (~30–40 MB of text)
  against 57M params will memorise early, and it is the slowest tier to run in-browser.
  A re-spec to **d=512 / L=8 / ctx=256 (~26M params, ~13 MB at 4-bit, ~2× faster
  inference)** is likely the better "large" tier at this data scale. Decide via eval
  (Phase 6) before locking the 768 numbers in.

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

**Caveats before the full 200K run:**
- This comparison predates the system prompt below. llama's reflexive
  question-flipping is exactly what the system prompt fixes, so the test was
  effectively *llama-without-system-prompt vs gemma*. **Re-run the 50-pair test
  as llama+system-prompt vs gemma+system-prompt** before committing — llama is
  3× faster, which is hours saved at 200K pairs if quality is now comparable.
- `gemma4-e2b-distill` is not a canonical Google release name (the `E2B`/`E4B`
  naming belongs to **Gemma 3n**, the on-device variant; there is no "Gemma 4").
  This is a local llama-swap alias — **record what GGUF / base model it actually
  resolves to** so the pipeline is reproducible later.

### System prompt architecture

The critical insight: using the OpenAI `system` role instead of embedding
instructions in user messages makes the model *obey* behavioral constraints.
Tested and validated base prompt:

```
You are a friendly, helpful AI assistant having a natural conversation.
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

**Length must be stratified, not fixed.** The previous prompt hard-coded "under
100 characters" — that contradicts the long-context goal: if every response is
<100 chars, Shade's 128 and Specter's 256 contexts are never filled and all three
tiers train on near-identical short pairs. Instead, sample a length instruction
per call from a distribution, so the corpus spans the full context range:

| Bucket | Share | Length instruction appended to system prompt |
|--------|-------|----------------------------------------------|
| Terse | 40% | "Reply in under 60 characters." |
| Short | 35% | "Reply in 1–2 sentences." |
| Medium | 20% | "Reply in 2–4 sentences with a little detail." |
| Long | 5% | "Reply in a short paragraph (3–5 sentences)." |

Wisp (ctx 64) trains mostly on Terse/Short; Shade and Specter draw the longer
buckets too. This is the single most important data change — it determines whether
EOS and the longer contexts train at all.

### Data expansion pipeline

Current: 2K pairs — fine for 3.3M params, way too little for 11M/57M.
Target: 5K+ pairs for Wisp retrain, 50K+ for Shade, 200K+ for Specter.

**Approach: Template-driven seed generation + gemma4-e2b-distill for responses**

Wisp is NOT used for prompt generation (tested — ~50% garbage at 3.3M/2K).
Instead, prompts come from deterministic templates, and gemma4-e2b-distill
provides all responses via the system-prompt pipeline above.

1. **Template seed bank** — hand-write ~200 prompt templates across categories:
   conversation, Q&A, jokes, recommendations, facts, creative, how-to, opinions,
   goodbyes, meta. Example: "Tell me a joke about {topic}",
   "How do I {verb} a {noun}", "What is the capital of {country}".
   **Drop code and math** — a byte-level model with ≤256 ctx cannot do either, and
   the teacher is weak at them; that data is noise for these students.

2. **Template expansion** — gemma4-e2b-distill expands 200 → 2,000 templates.
   Prompt: "Here are 5 chatbot prompt templates: [examples]. Generate 20 more
   on different topics. Output one per line." Parse defensively (strip numbering /
   preambles, reuse `batch_distill.clean()`); do not assume an exact count per call.

3. **Slot filling** — for each template, randomly sample from word lists
   (100 topics, 200 verbs, 150 countries, etc.) to produce 5K-10K unique seed
   prompts.

4. **Diversity pass** — for each seed prompt, gemma4-e2b-distill generates
   ~5 natural variations: "Given the prompt '{prompt}', generate 5 different
   ways a real user might phrase this." → 25K-50K distinct prompts. The "5×"
   is nominal — the model returns 3–6; **log actual yield** rather than assuming 50K.

5. **Response generation** — feed all prompts through gemma4-e2b-distill with the
   system prompt above, sampling a length bucket per call (see table). Collect
   prompt|response pairs.

6. **Multi-turn pass (Shade/Specter only)** — single short pairs waste a 256-token
   context. Generate a fraction of the data as 2–4 turn exchanges (teacher plays
   both sides, or continues a seeded thread), serialized with the separator token
   (see Data format). This is the most visible quality win for the larger tiers and
   the natural way to fill their context windows.

All intelligence steps (2, 4, 5, 6) run through gemma4-e2b-distill with
16-64 parallel workers, capped in practice by the single-GPU server's batching
(realistic effective concurrency ~4–8). Total calls:
- Step 2: ~90 batched calls (200 → 2,000 templates)
- Step 4: ~10K calls (one per seed prompt, ~5 outputs each)
- Step 5: ~50K calls (one per final prompt)
- **Total: ~60K calls. At the 50-pair test's 0.094 s/pair wall-clock, ~1.5–2h;
  budget a generous overnight to absorb retries and filtering.**

**Pick one canonical generation script.** `distill.py` has the system-prompt fix,
`batch_distill.py` has resume + checkpointing (but a hard-coded API-key path), and
Phase 2 calls for a new `expand_data.py`. Consolidate into one — otherwise the
system-prompt fix lives in the script you won't run at 200K scale.

### Data format & length budget

Decide these before generating 200K pairs — they are cheap now, expensive later:

- **Separator token.** Q and R are currently concatenated with no boundary. Add an
  explicit separator (a reserved token, or `\n`) between query and response. This
  makes the boundary learnable and is the foundation for multi-turn data.
- **Length budget, off-by-one.** Enforce `len(q) + len(r) + 1 (EOS) ≤ ctx` in the
  filter — "ctx/2 each" overflows (128+128+1 > 256). Apply the same filter when
  retraining Wisp.
- **EOS placement.** EOS must mark a *natural* end of response, never a mid-sentence
  truncation point. With the length filter above, EOS lands at real sentence ends and
  the model learns to self-terminate — which should let the UI's sentence-trim hack be
  removed.

### Training configs

**Budget by tokens, not epochs, and select on a validation split.** 200–400 epochs ×
50K pairs is ~1.5–3B tokens through an 11M model — heavy memorisation, far past
generalisation. With 25× more data than today you need *fewer* passes, not more.

- **Hold out a validation split (~2–5%)** and checkpoint on best *val* loss. The
  current loop checkpoints on best *train* accuracy, which systematically selects the
  most-overfit epoch.
- **Target ~10–40 epochs with early stopping on val loss**, not a fixed 200–400.
- **Mixed precision (bf16 autocast)** — the script is fp32-only today; bf16 roughly
  halves time/memory on the 5080.
- **Batch size is a real knob.** A 16 GB 5080 fits batch 64–256 at these sizes; the
  current loop hard-codes 16 and the plan's "batch 32 + grad accum 2/4" doesn't exist
  in code. Prefer a larger real batch over gradient accumulation at this scale.
- **QAT penalty cadence.** `compute_quantization_loss` runs `torch.quantile` on every
  parameter every step — compute it every N steps instead.

Per-tier starting points (revise from the val curve):

| Tier | pairs | ~epochs (early-stop) | lr | warmup | batch |
|------|-------|----------------------|-----|--------|-------|
| Wisp | 5K | 40–80 | 0.003 | 5 | 64 |
| Shade | 50K | 15–30 | 0.001 | 10 | 128 |
| Specter | 200K | 8–15 | 0.0005 | 20 | 64–128 |

Rough wall-clock on the 5080 with bf16 + larger batch: Shade ~1–2h, Specter ~6–12h.
(The earlier 8–16h Specter estimate assumed the fp32 / batch-16 / per-step-quantile
path; the figures above assume the fixes land first.)

### Sampling (inference)

Pure temperature over 258 logits is fine at Wisp scale, but tail-sampling errors
compound over 200+ token generations. Add **top-k / top-p** truncation for Shade and
Specter so a single low-probability byte can't derail a long response.

### Quantization & precision selection

Don't pre-decide Specter's precision — **train all three and measure the loss.**
Efficient path: train one fp32 reference, then QAT-finetune from that checkpoint to each
target (the model already trains quantization-aware via `compute_quantization_loss`).
This yields the fp32 ceiling for free and is cheaper than separate from-scratch runs.

- **fp32 reference** — quality ceiling (unshippable for Specter, but the baseline).
- **8-bit** (~57 MB Specter, ships) — expect near-lossless.
- **4-bit** (~28 MB Specter, ships) — expect some loss; depends on QAT + per-tensor scales.

Measure loss three ways, not by eye alone:
- **Val perplexity / byte-accuracy delta** vs fp32 on the held-out split — the clean number.
- **Logit divergence** (KL / MSE per token) from the parity harness — direct quant-noise metric.
- **Qualitative** — fixed eval prompt set, side-by-side.

Ship the smallest precision whose loss is acceptable. **Run the comparison on Shade first**
(cheap to train and quantize) to establish methodology and the expected loss curve before
spending the Specter budget. Needs a `--weight-bits {4,8,32}` knob on the trainer and the
canonical serializer so quantizer range + packing follow one flag.

## Implementation checklist

Phases are dependency-ordered. Model **quality** is evaluated in PyTorch on the GPU
throughout (fast; no browser needed), so nothing below gates evaluation. KV cache
(Phase 3) is what makes Specter *usable in-browser* — needed before Specter ships
(Phase 6), an immediate UX win for Wisp, not a blocker for anything earlier. The
quantization path (Phase 5) gates Specter shipping at all.

### Phase 1: Infrastructure
- [ ] Canonical serializer: one script, any arch (via manifest), with a recorded
      output format. (The shipped bundle currently has no source script — fix first.)
- [ ] **Parity harness**: Python forward-pass logits vs Node-run TS/Wasm logits on
      fixed prompts, tolerance-checked, wired into CI. This is the defence against
      serializer bugs, for all three models.
- [ ] TS orchestrator: fully arch-driven (vocab_size / max_len from manifest, no
      hard-coded dims), scratch margin computed from arch.
- [ ] Wasm kernel: **needs a per-tensor `scale` parameter on the packed matmul** for
      the 4-bit path (the current hard-coded `GLOBAL_WEIGHT_SCALE` and the three
      inconsistent scale schemes must be unified). Not "no changes needed."
- [ ] `models.json` registry (name, files, byte size, content hash) driving the UI;
      hashed filenames instead of hand-bumped `?v=N`.
- [ ] Dist: `.bin`/`.json` per model (`model_wisp.*`, `model_shade.*`, `model_specter.*`).

### Phase 2: Data expansion
- [ ] Consolidate to one canonical `expand_data.py` (fold in distill.py system prompt + batch_distill resume).
- [ ] Hand-write 200 prompt templates across categories (no code/math).
- [ ] Template expansion via teacher (200 → 2,000 templates), defensive parsing.
- [ ] Slot-filling with word lists → 5K–10K seed prompts.
- [ ] Diversity pass: ~5 variations per seed → 25K–50K prompts; log actual yield.
- [ ] Response generation with length-stratified system prompt.
- [ ] Multi-turn pass for Shade/Specter (separator token).
- [ ] Filter: dedupe (exact + near-dup), non-ASCII strip, `len(q)+len(r)+1 ≤ ctx`,
      reject refusals / teacher preambles.
- [ ] Validate: quick Wisp retrain on expanded data to check quality.

### Phase 3: Inference performance — KV cache
Not a prerequisite for *evaluating* model quality (do that in PyTorch). This is the gate
for Specter being **usable in-browser**: without it Specter is minutes-per-response
(~18× Wisp's per-position cost, O(T²) over a 4× longer context — far past the ~3 s a
Wisp generation takes). Plus an immediate UX win for Wisp. Land it any time before
Specter ships (Phase 6).
- [ ] Per-layer K/V cache; per step compute Q/K/V + FFN for the new position only
      (turns O(T²) per response into O(T)).
- [ ] Validate on Wisp (immediate UX win).
- [ ] (Follow-on) move attention dot-product loops from JS into the Wasm kernel; SIMD.

### Phase 4: Training (small tiers)
- [ ] Add validation split + early-stopping on val loss; bf16 autocast; `--batch-size`.
- [ ] Retrain Wisp on expanded, filtered data (baseline; validates the whole pipeline).
- [ ] Train Shade; serialize (float32 acceptable for first build) + parity check.
- [ ] Evaluate Shade vs Wisp on a fixed 10–20 prompt eval set.

### Phase 5: Quantization path (4-bit + 8-bit) — gate for Specter
- [ ] `--weight-bits {4,8,32}` on trainer + canonical serializer (range/packing/scales follow the flag).
- [ ] Kernel: per-tensor `scale` param + 8-bit and 4-bit unpack paths (from Phase 1).
- [ ] Mixed-precision layout: quantize attention/FFN matmuls; keep embeddings, head,
      LN, biases in float32 (sensitive, and cheap — ~1.6 MB for Specter).
- [ ] Align the QAT penalty to the per-tensor scheme, parametrized by weight-bits.
- [ ] On Shade: fp32 + 8-bit + 4-bit; measure loss (val ppl, logit KL, qualitative).
      Establishes methodology before Specter.
- [ ] Re-serialize the chosen Wisp/Shade builds; parity-check vs fp32 within tolerance.

### Phase 6: Specter
- [ ] Decide d=768 vs re-spec to d=512 from an ablation on the new data (don't pre-decide).
- [ ] Train fp32 reference → QAT-finetune 8-bit and 4-bit.
- [ ] Evaluate quality loss (4-bit vs 8-bit vs fp32); ship the smallest precision that
      holds — 8-bit (~57 MB) if 4-bit (~28 MB) degrades too far.
- [ ] KV cache (Phase 3) must be in before Specter is usable in-browser; confirm tok/s.

### Phase 7: UI
- [ ] Model switcher in terminal (tabs or dropdown), driven by `models.json`.
- [ ] Default: Wisp (loads immediately); Shade/Specter load on demand.
- [ ] Show model name + size in status bar; loading progress for larger models.
- [ ] Cache loaded models in memory (don't re-download on switch).
- [ ] top-k / top-p sampling controls for the larger tiers.

### Phase 8: Polish & CI
- [ ] Boot sequence shows available models (no hard-coded "256d × 4L | 13MB").
- [ ] Graceful fallback if a model fails to load.
- [ ] CI ships committed `.bin`s (it cannot train); enforce every file ≤ 100 MB.
- [ ] Parity harness (Phase 1) runs in CI on every model.
