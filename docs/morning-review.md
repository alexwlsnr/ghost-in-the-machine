# Morning Review — Overnight Branch Summary

All branches off `feature/multi-model`. Nothing merged. Review each and decide
merge order before touching main.

---

## Suggested merge order (dependency-first)

```
feature/trainer-integration        # trainer: val-split + supervision + preserve-case
  └── feature/expand-data          # Phase 2 data pipeline
        └── feature/templates-and-scripts-2  # 200 templates + launch scripts
feature/sampling                   # top-k/top-p sampling (pure TS, no deps)
feature/model-switcher             # UI: models.json + dynamic switcher
feature/quantization               # 4-bit/8-bit kernel + serializer (gates Specter)
feature/kv-cache                   # O(T²)→O(T) inference (needed before Specter ships)
feature/wraith-mvp                 # Linux guru MVP (depends on preserve-case in trainer)
```

Analysis branches (docs only, no code changes — read then delete):
- `analysis/bonsai-model4` — **No Whisper tier now.** Size savings negligible at <1M params,
  no Wasm ternary CPU inference path. Revisit after Specter + SIMD128.
- `analysis/linux-guru-model` — **Wraith-C recommended.** d=384, 8L, ctx=256, 14.5M params,
  ~7.2MB at 4-bit. Mean useful Linux answer: 71 bytes. tldr-pages corpus: 7,243 pages CC0.
- `analysis/ts-wasm-split` — **KV cache #1**, then attention_f32 Wasm export, then 4-bit.
  JS attention is ~45% of wall-clock at T=255 despite being <5% of FLOPs (interpreter overhead).

---

## Branch details

### ✅ `feature/trainer-integration`
**Combines feature/train-validation + feature/supervision-harness, plus --preserve-case.**
The canonical trainer branch. All subsequent training work starts from here.

Changes vs `feature/multi-model`:
- `split_pairs()`, `_build_sequences()` helpers
- `train_transformer()`: `val_frac`, `patience`, `status_file`, `preserve_case` params
- `generate()`: `preserve_case` param
- CLI: `--val-frac`, `--patience`, `--status-file`, `--preserve-case`
- `py/training_status.py`: `write_status()` / `read_status()`, atomic fsync+rename
- `scripts/watch_training.sh`: pure-bash watcher, exits on events + 45-min heartbeat
- `scripts/train_shade.sh`: nohup detach pattern + initial status write

Tests: 33 Python (18 train loop + 15 status emitter), 12 JS — all pass.

⚠️ **Note:** `feature/train-validation` is superseded by this branch. Delete it after merge.

---

### ✅ `feature/expand-data`
**Phase 2 data generation pipeline (`py/expand_data.py`).**

5-phase pipeline: template seed bank → template expansion → slot filling →
diversity pass → response generation (length-stratified). Resume-safe checkpointing.
Teacher model CLI flag (default `gemma4-e4b-distill`). `--max-ctx` filter
(`len(q)+len(r)+1 ≤ ctx`) also fixes the EOS-truncation bug (#5 from pre-work).

Tests: 64 tests across 8 classes (pure functions only — no teacher calls).

---

### ✅ `feature/templates-and-scripts-2`
**200 prompt templates + complete launch scripts.**

- `data/templates.txt`: 200 templates across 9 categories (conversation 30, Q&A 30,
  jokes 20, recommendations 25, how-to 25, opinions 20, creative 20, goodbyes 15,
  meta 15). No code/math.
- `scripts/train_wisp.sh`, `train_shade.sh`, `train_specter.sh`: correct hyperparams
  per tier, preserve-case off (conversational models).
- `scripts/run_data_gen.sh`: wraps expand_data.py with defaults; respects
  `$EXPAND_MODEL` / `$EXPAND_ENDPOINT` env vars.

⚠️ Note: scripts reference `--val-frac`, `--patience`, `--status-file` which are
in `feature/trainer-integration`. Merge that first.

---

### ✅ `feature/sampling`
**Top-k / top-p nucleus sampling in the TS orchestrator.**

- `sampleFromLogits(logits, temp, topK, topP, rand)`: pure function, replaces inline
  sampling in `generate()`.
- `generate()` gains `topK=0` and `topP=1.0` params (defaults preserve existing behaviour).
- Recommended for Shade/Specter: `topK=40, topP=0.9`.
- Also includes the quantization-dispatch matmulWeights helper from the quantization
  agent's working tree (bonus, doesn't affect fp32 model behaviour).

Tests: 8 new sampling tests, full suite 26/26.

---

### ✅ `feature/model-switcher`
**UI: models.json registry + dynamic model switcher.**

- `dist/models.json`: 3-model registry (Wisp available, Shade/Specter "coming soon").
- `dist/model_wisp.bin` / `model_wisp.json`: renamed Wisp files (originals kept).
- `dist/index.html`: model chip bar, active model highlighting, cached switching,
  dynamic boot sequence, status bar shows "WISP · 3.3M params · 64 ctx".
- Known: no per-byte loading progress (loadModel has no progress callback).

---

### ✅ `feature/supervision-harness`
**Superseded by `feature/trainer-integration`.** Already incorporated. Delete after merge.

---

### ✅ `feature/wraith-mvp`
**Linux guru model MVP scaffold.**

- `py/train_transformer.py`: `--preserve-case` flag (also in trainer-integration).
- `py/wraith_data.py`: fetches tldr-pages from GitHub, parses into Q|R pairs.
  Results: 346/359 commands found, **4,720 training pairs** generated.
- `scripts/train_wraith.sh`: Wisp-scale MVP (d=256, 4L, ctx=128, preserve-case).
- `data/wraith_eval.txt`: 20 held-out Linux Q&A pairs for evaluation.
- `test/test_preserve_case.py`: 10 tests passing.

MVP path: run `scripts/train_wraith.sh`, score against `data/wraith_eval.txt`.
≥15/20 → scale to Wraith-C (d=384, 8L, ctx=256).

⚠️ Merge `feature/trainer-integration` first (preserve-case is in both, trainer-int
is the authoritative version).

---

### 🔄 `feature/kv-cache` (may not be complete yet)
**O(T²)→O(T) per-token attention via KV cache.**

If pushed: `createCache()`, `forwardIncremental()`, `prefill()`, cache-aware `generate()`.
Correctness test: same prompt + same seed → identical output with/without cache.
Critical for Specter (without it: ~2s/token in-browser).

---

### 🔄 `feature/quantization` (may not be complete yet)
**4-bit + 8-bit kernel and serializer — gates Specter shipping.**

If pushed:
- `wasm/src/lib.rs`: `matmul_4bit(scale)` and `matmul_8bit(scale)` exports (per-tensor
  scale, no hardcoded global constant).
- `py/serialize.py`: `--weight-bits 32/8/4`, mixed-precision (quantize attn/FFN matrices
  only, keep embeddings/head/LN/biases fp32).
- `ts/src/tier2_transformer.ts`: `matmulWeights` dispatch on manifest dtype.

---

## What to run in the morning

```bash
# 1. Merge in order (resolve conflicts as noted)
git checkout feature/multi-model
git merge feature/trainer-integration  # base for everything
git merge feature/expand-data
git merge feature/templates-and-scripts-2
git merge feature/sampling
git merge feature/model-switcher
git merge feature/quantization         # if ready
git merge feature/kv-cache             # if ready
git merge feature/wraith-mvp

# 2. Start data generation (run in a terminal, takes ~1-2h at 32 workers)
bash scripts/run_data_gen.sh

# 3. Once data is ready, train Wisp (validate the pipeline)
bash scripts/train_wisp.sh --data data/training_pairs.txt

# 4. Wraith MVP (can run in parallel with Wisp)
bash scripts/train_wraith.sh

# 5. Check Wisp quality against generation goldens
npm test
```

## Test counts (at time of writing)

| Suite | Tests | Status |
|---|---|---|
| JS (kernel, parity, generation, forward_arch, sampling) | 26 | ✅ all pass |
| Python (train_loop, generate, training_status, expand_data, preserve_case) | 95+ | ✅ all pass |
