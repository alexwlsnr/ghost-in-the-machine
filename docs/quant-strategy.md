# Quantization & Tier Strategy

_Ghost in the Machine — revised June 2026_

## TL;DR

1. **Ship Wisp at 4-bit** (2.1 MB, ~100 ms load). The per-tensor scale bug is fixed;
   QAT conditioning should hold quality within acceptable bounds. Validate after the first
   re-train on expanded data.
2. **Redesign tiers around download budgets**, not parameter counts. The current table
   has a design flaw where Shade's 4-bit artifact (5.5 MB) is _smaller_ than Wisp's fp32
   (13 MB), which punishes Shade's quality for no shipping benefit.
3. **Keep Shade as-is for Tier 2; make Spec512 the canonical Tier 3** (d=512, 8L,
   ctx=256). Specter (d=768) only if Spec512 eval is disappointing and the budget can
   absorb the extra training time.
4. **Train in order: Wisp → Shade → Spec512.** The full quant experiment for all three
   tiers costs under 3 hours of GPU time at measured bf16 throughput.

---

## 1. Parameter counts and sizes (corrected)

The multi-model plan table uses simplified full-quantization sizing. The canonical
serializer (`py/serialize.py`) applies **mixed-precision**: embeddings, head, layer norms
and biases stay fp32; only attention (Q/K/V/O) and FFN weight matrices are quantized.
Corrected figures using that layout:

### Formula reference

```
vocab = 258, d_head = 64 (always)
embed       = vocab·d + ctx·d
per_layer   = 4·(d·d) + 4·d          # attn weights+biases (Q,K,V,O)
            + d·dff + dff             # ff1 weight + bias
            + dff·d + d               # ff2 weight + bias
            + 4·d                     # 2 LN (weight+bias) × 2
head        = vocab·d
total       = embed + L·per_layer + 2·d + head
```

Mixed-precision split (per tier, approximately):
- **fp32-only** (embed + head + LN + biases): small, ~100–700 KB across all tiers
- **quantizable** (attn + FFN weight matrices): ~95% of total params

### Corrected size table

| Arch | d | L | ctx | dff | Params | fp32 | 8-bit | 4-bit | ternary* |
|------|---|---|-----|-----|--------|------|-------|-------|---------|
| **Wisp** | 256 | 4 | 64 | 1024 | 3.3M | 12.6 MB | 3.6 MB | **2.1 MB** | 1.2 MB |
| **Shade** | 384 | 6 | 128 | 1536 | 10.9M | 41.6 MB | **11.2 MB** | 6.1 MB | 3.1 MB |
| **Spec512** | 512 | 8 | 256 | 2048 | 25.6M | 97.7 MB | 25.7 MB | **13.7 MB** | 6.5 MB |
| Specter | 768 | 8 | 256 | 3072 | 57.3M | 218.6 MB | 56.6 MB | 29.6 MB | 13.2 MB |

_* ternary = 1.58 bits/weight; experimental — training-time weight constraint not yet
implemented. See section 2._

The plan's previous "Shade 4-bit ~5.5 MB" figure assumed full quantization of all
weights, including embeddings. Corrected mixed-precision 4-bit is 6.1 MB; 8-bit is
11.2 MB. Similarly Specter fp32 is 218.6 MB (not 219 MB — the 219 figure in the plan is
close enough and not a concern).

---

## 2. Revised tier table — fixed size buckets

Design principle: choose the precision that maximises parameter count within a hard
download budget. Quality ceiling is always the fp32 reference; the question is how far
quantization degrades it.

### Budget targets

| Tier | Budget | Load feel | Target |
|------|--------|-----------|--------|
| 1 | ~2–3 MB | Instant, in-page | Ships before user notices |
| 2 | ~15–20 MB | Fast (~2–4 s on LTE) | Load on first select |
| 3 | ~50–60 MB | Acceptable (~10–15 s on LTE) | Load on explicit request |

### Tier 1 — Instant (~2–3 MB)

| Candidate | Params | Size | Notes |
|-----------|--------|------|-------|
| Wisp fp32 | 3.3M | 12.6 MB | Over budget |
| Wisp 8-bit | 3.3M | 3.6 MB | Fits, but wastes the budget advantage |
| **Wisp 4-bit** | **3.3M** | **2.1 MB** | Recommended — see section 3 |

**Winner: Wisp 4-bit (2.1 MB).** After the per-tensor scale fix, this is viable. The
0.4 MB sensitivity overhead (fp32 embeds/head) is already included.

### Tier 2 — Fast (~15–20 MB)

| Candidate | Params | Size | Notes |
|-----------|--------|------|-------|
| Shade fp32 | 10.9M | 41.6 MB | Over budget |
| Shade 8-bit | 10.9M | 11.2 MB | Fits; 3.3× more params than Wisp at near-lossless quality |
| Shade 4-bit | 10.9M | 6.1 MB | Fits; leaves 10 MB headroom but that's not a problem |
| **Spec512 4-bit** | **25.6M** | **13.7 MB** | Fits; 7.8× more params than Shade 4-bit |
| Spec512 8-bit | 25.6M | 25.7 MB | Slightly over 20 MB, but under 30 MB; decide by eval |

Two credible options for Tier 2:

- **Option A (conservative): Shade 8-bit** — 10.9M params, 11.2 MB, near-lossless.
  Natural intermediate between Wisp (3.3M) and Spec512 (25.6M). Good stepping stone
  if Spec512 requires substantially more training data to shine.
- **Option B (aggressive): Spec512 4-bit** — 25.6M params, 13.7 MB. Fits the budget
  cleanly and provides 7.8× more capacity than Shade 4-bit. The risk: 25M params
  trained on 50K pairs may not beat 11M params at 8-bit, since the larger model has
  more underutilised capacity. This is an empirical question — run the eval.

**Recommendation: start with Option A (Shade 8-bit); run Option B as a stretch target
after Shade eval shows whether the architecture bottleneck is params or data.** Both
sizes are already shippable.

### Tier 3 — Acceptable (~50–60 MB)

| Candidate | Params | Size | Notes |
|-----------|--------|------|-------|
| Spec512 fp32 | 25.6M | 97.7 MB | Over budget, unshippable |
| Spec512 8-bit | 25.6M | 25.7 MB | Fits, but wastes the size budget |
| Specter fp32 | 57.3M | 218.6 MB | Unshippable (exceeds GitHub 100 MB limit) |
| **Specter 8-bit** | **57.3M** | **56.6 MB** | Recommended — near-lossless, maximises params |
| Specter 4-bit | 57.3M | 29.6 MB | Fits with headroom; 4-bit loss is a concern at 57M |

**Winner: Specter 8-bit (56.6 MB).** This is right at the 50–60 MB budget, maximises
params, and 8-bit is well-established as near-lossless at this scale. Specter 4-bit
(29.6 MB) is a viable fallback if quality holds, and saves 27 MB download time. Decide
empirically: train fp32 reference, QAT-finetune at 8-bit and 4-bit, compare eval.

**Caveat — d=768 vs d=512:** Specter (d=768, 57M params) on 200K pairs may memorise
rather than generalise. Spec512 (d=512, 25.6M params) is 2× faster per step, fits the
same context, and may produce _better_ generalisation at this data scale. The morning
review already flagged d=512 as the preferred "large" tier at 200K pairs. **Train
Spec512 first; escalate to Specter only if eval on the Spec512 plateau is clearly
insufficient.**

### Ternary (1.58-bit) — experimental rows

At 1.58 bits/weight, a 50 MB budget holds ~265M params. That is beyond the current
architecture scope, but ternary is included for completeness at the planned scales:

| Arch | Ternary size | Notes |
|------|-------------|-------|
| Wisp | 1.2 MB | Sub-optimal: 3.3M params too small for ternary — see below |
| Shade | 3.1 MB | Interesting but no training support yet |
| Spec512 | 6.5 MB | 25.6M params at 1.58-bit; data efficiency risk same as 4-bit |
| Specter | 13.2 MB | Compelling — 57M params in 13 MB |

**Ternary viability assessment (from `analysis/bonsai-model4`):**
- Sub-1M scale: not viable. Weights constrained to {-1, 0, +1} collapse representational
  power below the threshold needed for coherent language modelling. The analysis branch
  found no quality path at these sizes.
- 3–11M scale (Wisp/Shade): marginal. The QAT loss already biases weights toward
  quantisation-friendly distributions, but 16 levels (4-bit) is meaningfully better than
  3 levels (ternary) at this capacity range. 4-bit is strictly preferred.
- 50M+ scale (Specter): compelling. At large scale, ternary benefits from the same
  "scale compensates for reduced precision" dynamic that makes 1-bit networks viable in
  BitNet-style architectures. 13.2 MB for 57M params is a genuinely attractive target.

**Blocker:** ternary requires a weight constraint to {-1, 0, +1} applied during training
(typically via `sign(w)` in the forward pass and a straight-through estimator for
gradients). The current `compute_quantization_loss()` pushes toward integer-friendly
distributions but does not enforce the ternary constraint. Implementing this is Phase 5.5
work — not a current blocker. Flag all ternary rows as **experimental** in `models.json`.

---

## 3. Wisp quantization analysis

### The previous problem

Before the `feature/quantization` branch, the Wasm kernel used a hardcoded
`GLOBAL_WEIGHT_SCALE = 0.4` regardless of the actual weight distribution. This caused
4-bit inference degradation that was not a fundamental limitation of 4-bit quantisation
— it was a bug. The fix (per-tensor scale stored per-section in the manifest as
`section.scale`) eliminates the root cause.

### Expected quality at each precision

**Wisp fp32 (12.6 MB) — quality ceiling**
- Full 32-bit weight resolution. No quantisation artefacts.
- Too large for the "instant" tier goal; loads in ~500–800 ms from a CDN.
- Keep as the reference checkpoint only; do not ship as default.

**Wisp 8-bit (3.6 MB) — near-lossless**
- 8-bit integer: 256 discrete levels per weight.
- Literature consensus: <1% accuracy loss vs fp32 at 3M+ params with QAT.
- Scale = max_abs / 127.0 per tensor (current serializer). Well-studied.
- The QAT penalty in `compute_quantization_loss()` already pushes weights toward
  quantisation-friendly distributions, so trained-with-QAT 8-bit should show
  negligible degradation.
- Verdict: viable, but 3.6 MB vs 2.1 MB for 4-bit is a real cost at this tier.

**Wisp 4-bit (2.1 MB) — recommended**

The key question: does 4-bit (16 levels per weight) hurt a 3.3M-param model?

Arguments for concern:
- 3.3M params is extremely small for a language model. Every weight carries
  proportionally more information than in a large model, making precision loss more
  impactful per parameter.
- Post-4-bit, the effective information density per weight drops from ~8 bits (fp32 ≈ 7
  bits of meaningful precision for these weight magnitudes) to 4 bits — a 2× reduction
  in representational fidelity.
- Literature estimates for _uncompensated_ 4-bit quantisation at 3M param scale:
  5–15% accuracy degradation, potentially more for sensitive layers (embedding, head).

Arguments for viability after the fix:
- Mixed-precision layout keeps embeddings and the output head at fp32. These are the
  most quantisation-sensitive tensors (embedding lookup is a one-hot selection;
  output head small errors compound across the full vocabulary). The quantised 95% is
  only the inner weight matrices.
- QAT from scratch pushes all inner weights toward quantisation-friendly distributions
  during training, not as post-hoc retrofit. Training-time QAT at 4-bit is materially
  better than PTQ (post-training quantisation) at 3M scale.
- The per-tensor scale (max_abs / 7.0) ensures optimal dynamic range per tensor rather
  than a shared global scale that under- or over-fits different layers.
- Wisp is already capacity-constrained at 3.3M params/5K pairs. The quality ceiling
  is low enough that 4-bit degradation may be imperceptible relative to the model's
  inherent limits.

**Recommendation: ship Wisp 4-bit as the primary artifact.** The risk of meaningful
quality regression is low given the mixed-precision layout, QAT conditioning, and per-
tensor scales. Validate empirically: after re-training on expanded data, run the fixed
20-prompt eval set in parallel at fp32, 8-bit, and 4-bit; if 4-bit shows visible
degradation (wrong word choices, garbled tokens), fall back to 8-bit (3.6 MB).

If 8-bit is required: it is still well within the spirit of the "instant" tier at 3.6 MB.
The regression threshold should be a _qualitative_ one (responses that are noticeably
wrong), not a raw perplexity threshold, since perplexity at this scale can be misleading.

---

## 4. Training time table

Measured throughput (bf16, RTX 5080 16 GB):

| Model | d | L | ctx | Batch | Measured steps/s |
|-------|---|---|-----|-------|-----------------|
| Wisp | 256 | 4 | 64 | 128 | 323 |
| Shade | 384 | 6 | 128 | 128 | 35 |
| Spec512 | 512 | 8 | 256 | 128 | 7.0 |
| Specter | 768 | 8 | 256 | 96 | 5.3 |

### fp32 reference training

| Model | Pairs | Steps/epoch | Epoch range | Step range | Wall-clock range |
|-------|-------|------------|-------------|------------|-----------------|
| Wisp | 5K | 40 | 40–80 | 1,600–3,200 | 5–10 s |
| Shade | 50K | 391 | 15–30 | 5,865–11,730 | 2.8–5.6 min |
| Spec512 | 200K | 1,563 | 8–15 | 12,504–23,445 | 30–56 min |
| Specter | 200K | 2,084 | 8–15 | 16,672–31,260 | 52 min–1.6 h |

_Note: Wisp training is nearly instant on the 5080. Elapsed time for the full
fp32-ref + 8-bit + 4-bit experiment is dominated by data generation, not training._

### QAT finetune from fp32 checkpoint

Finetune for ~25% of the maximum epoch count with a reduced lr warmup from the fp32
weights. This is cheaper than full re-training and yields a better quantised model than
PTQ because the weights adapt to quantisation noise rather than merely being rounded.

| Model | QAT epochs | Steps | Wall-clock per precision |
|-------|-----------|-------|--------------------------|
| Wisp | 20 | 800 | ~2 s |
| Shade | 7 | 2,737 | ~1.3 min |
| Spec512 | 4 | 6,252 | ~15 min |
| Specter | 4 | 8,336 | ~26 min |

### Full quant experiment (fp32 ref + 8-bit finetune + 4-bit finetune)

| Model | fp32 ref | + 8-bit QAT | + 4-bit QAT | Total |
|-------|----------|-------------|-------------|-------|
| Wisp | ~10 s | ~2 s | ~2 s | ~15 s |
| Shade | ~5.6 min | ~1.3 min | ~1.3 min | ~8 min |
| Spec512 | ~56 min | ~15 min | ~15 min | ~1.4 h |
| Specter | ~1.6 h | ~26 min | ~26 min | ~2.5 h |

**Total GPU time for all four models (fp32 + 8-bit + 4-bit):** approximately 4 hours,
all on the RTX 5080. Data generation (~1.5–2 h for 50K pairs, longer for 200K) is the
real bottleneck, not training.

### Important notes on these estimates

- Epoch ranges include early stopping: the wall-clock "range" assumes stopping at
  min–max epochs; real runs with a good val curve will stop toward the lower end.
- The `compute_quantization_loss()` penalty runs every `--qat-every N` steps. At the
  default of every step it is expensive (per-tensor `torch.quantile`). Set
  `--qat-every 10` to reduce overhead ~5–10× with negligible quality impact.
- QAT finetune sessions should start from the best-checkpoint fp32 weights, not the
  final epoch, to avoid fine-tuning from an overfit snapshot.

---

## 5. Experiment plan

### Phase A — Pipeline validation (Wisp)

_Goal: confirm the whole pipeline (data → train → serialize → parity → eval) works
before spending 200K-pair generation budget on Shade/Spec512._

1. Merge `feature/quantization` into `feature/multi-model` (all tests must pass;
   re-serialize after to update manifest format).
2. Run `scripts/run_data_gen.sh` for 5K pairs (Wisp scale). Estimated: ~15 min at 16
   workers.
3. Train Wisp fp32 reference:
   ```bash
   python3 py/train_transformer.py --file data/training_pairs.txt \
     --d-model 256 --n-layers 4 --d-ff 1024 --max-len 64 \
     --epochs 80 --lr 0.003 --batch-size 128 --amp \
     --val-frac 0.05 --patience 10 \
     --qat-every 10 --qat-weight 0.10 \
     --checkpoint ckpt/wisp_fp32.pt
   ```
4. Serialize at all three precisions:
   ```bash
   python3 py/serialize.py ckpt/wisp_fp32.pt --out dist/model_wisp --weight-bits 32
   python3 py/serialize.py ckpt/wisp_fp32.pt --out dist/model_wisp_8 --weight-bits 8
   python3 py/serialize.py ckpt/wisp_fp32.pt --out dist/model_wisp_4 --weight-bits 4
   ```
   Note: serializing from a single fp32 checkpoint is PTQ. For better 4-bit quality,
   follow with QAT finetune (step 5).
5. QAT finetune for 8-bit and 4-bit (optional but recommended):
   ```bash
   # 4-bit finetune
   python3 py/train_transformer.py --file data/training_pairs.txt \
     --resume ckpt/wisp_fp32.pt \
     --epochs 20 --lr 0.0005 --batch-size 128 --amp \
     --val-frac 0.05 --patience 5 \
     --qat-every 1 --qat-weight 0.20 \
     --checkpoint ckpt/wisp_4bit.pt
   ```
6. Evaluate: run the 20-prompt eval set (extend `data/wraith_eval.txt` pattern for
   conversational prompts) at fp32, 8-bit PTQ, and 4-bit QAT. Record qualitative
   ratings.
7. **Decision gate:** if 4-bit QAT passes qual threshold, set as Wisp default artifact.
   If not, 8-bit. If 8-bit fails, investigate before proceeding.

### Phase B — Shade quant comparison

_Goal: establish the quant methodology on a mid-size model before committing to
200K-pair generation._

1. Generate 50K pairs with length stratification (run `expand_data.py` with Shade
   ctx=128 filter). Estimated: ~1.5 h at 16 workers.
2. Train Shade fp32 reference (~5.6 min).
3. QAT finetune at 8-bit and 4-bit (~1.3 min each).
4. Serialize at fp32/8-bit/4-bit.
5. Evaluate: val perplexity delta vs fp32, logit KL on fixed prompts, qualitative.
6. **Decision gate:** choose the smallest precision that stays within acceptable quality
   loss. This also answers whether Spec512 4-bit is competitive vs Shade 8-bit.

### Phase C — Spec512 (primary large tier)

1. Reuse the 200K pairs generated for Specter (or generate a Spec512-specific set with
   ctx=256 filter). Estimated: ~2–3 h for 200K pairs.
2. Train Spec512 fp32 reference (~56 min worst case; likely ~30–40 min with early stop).
3. QAT finetune at 8-bit and 4-bit (~15 min each).
4. Evaluate vs Shade on quality and inference speed.
5. **Decision gate:** if Spec512 clearly outperforms Shade → promote to Tier 3.
   If Spec512 ≈ Shade quality → Shade becomes Tier 2, Spec512 is optional.

### Phase D — Specter (optional large tier, only if Spec512 is insufficient)

_Proceed only if Phase C eval shows Spec512 has hit a quality ceiling that more params
would overcome. Specter at 57M params on 200K pairs risks memorisation._

1. Train Specter fp32 reference (~1.6 h worst case).
2. QAT finetune 8-bit (~26 min).
3. 4-bit finetune only if 8-bit passes qual at 8-bit (56.6 MB).
4. If 4-bit (29.6 MB) also passes → ship 4-bit to stay within the 50–60 MB budget.

### Evaluation harness

Define a fixed conversational eval set (`data/ghost_eval.txt`) of 20–30 prompts across
categories: greetings, jokes, factual Q&A, recommendations, opinions, creative tasks.
For each precision under test, score:

| Metric | Method | Threshold |
|--------|--------|-----------|
| Val perplexity delta | `(ppl_quant - ppl_fp32) / ppl_fp32` | < 5% |
| Byte accuracy delta | Held-out split exact match | < 2 pp drop |
| Logit KL divergence | `KL(fp32 logits ‖ quant logits)` per token | < 0.05 nats |
| Qualitative | Fixed 20-prompt set, side-by-side comparison | No clearly wrong responses |

Ship the smallest precision whose _all four_ metrics pass. KL divergence is the most
sensitive early signal; qualitative is the user-facing gate.

---

## 6. Recommended action order

Given that data generation is running now (~50K Wisp/Shade scale, 200K Specter scale):

### Immediate (today / this session)

1. **Merge `feature/quantization`** — required for 4-bit/8-bit serialization. After
   merging, re-serialize the existing Wisp checkpoint at fp32 to update the manifest
   format (`py/serialize.py --weight-bits 32`), regenerate parity fixtures
   (`npm run fixtures`), confirm all tests pass.

2. **Train Wisp on expanded 5K dataset** as a pipeline validation. Expected: ~10 s.
   This is the mandatory smoke test before investing in larger models.

3. **Run the Wisp quant comparison** (fp32 / 8-bit PTQ / 4-bit QAT) while waiting for
   Shade data to finish generating. Cost: ~15 s total GPU time. Decision determines the
   Wisp production artifact.

### Short term (once 50K pairs are ready)

4. **Train Shade fp32 + QAT finetune** (~8 min total). This is the quant methodology
   calibration step — if 4-bit Shade passes eval, the same approach scales to Spec512.

5. **Decide Tier 2 architecture**: if Shade 8-bit at 11.2 MB passes qual, ship that.
   If 4-bit Shade also passes, consider whether to try Spec512 4-bit (13.7 MB, 25.6M
   params) as an upgrade. A/B test on the eval set.

### Medium term (once 200K pairs are ready)

6. **Train Spec512 fp32 + QAT finetune** (~1.4 h total). Compare on the eval set.
   This is the candidate Tier 3 model.

7. **Specter only if Spec512 eval is disappointing.** Budget ~2.5 h GPU + the risk of
   overfitting 57M params on 200K pairs. If pursued: train fp32 reference, then 8-bit
   QAT finetune; 4-bit only if 8-bit passes at acceptable quality.

### Training approach: QAT from scratch vs finetune

**Recommendation: QAT-finetune from fp32 checkpoint, not QAT from scratch.**

Reasons:
- The fp32 reference is nearly free to train (minutes for Wisp/Shade, under an hour for
  Spec512). It yields the quality ceiling and a warm-started checkpoint.
- QAT finetune from fp32 is substantially better than PTQ (serialising a fp32 checkpoint
  at 4-bit without any training) because the weights can adapt to quantisation noise.
- QAT from scratch produces similar final quality to finetune in most literature at this
  scale, but requires a full training run per precision. Finetune amortises the cost.
- Exception: if the fp32 reference shows poor generalisation (val loss not converging),
  consider QAT from scratch for the 8-bit model with a higher `--qat-weight` (0.15–0.20)
  from epoch 1.

### When to run the quant comparison experiment

- **Wisp:** immediately after the expanded-data retrain. Cost is negligible.
- **Shade:** after 50K pairs are ready and Shade fp32 is trained. Confirms methodology.
- **Spec512:** after the Shade comparison, once 200K pairs are ready. Builds confidence.
- **Specter:** only if Phase C decision gate says it is needed.

---

## 7. Open questions

1. **Does Shade 4-bit (6.1 MB) offer quality comparable to Shade 8-bit (11.2 MB)?**
   Both fit in the Tier 2 budget. If they are equivalent, ship 4-bit and save 5 MB.
   Answer via Phase B eval.

2. **Is Spec512 on 200K pairs data-efficient enough to beat Shade on quality?**
   At 25.6M params / 200K pairs, the params-per-pair ratio is 128:1 — similar to Shade
   at 10.9M / 50K (218:1). More data per param for Spec512, which is a good sign.

3. **Does Specter (57M params, 200K pairs) generalise or memorise?**
   57M params / 200K pairs = 285:1. Shade ran at 218:1. These are close enough that
   Specter may generalise, but the model is far larger than Wisp (which trained at 660:1
   on 5K pairs). The plan's concern about early memorisation is legitimate — monitor val
   loss carefully, and early-stop aggressively.

4. **What quality threshold justifies a 10-second download for Tier 3?**
   Users who select a larger model should see a clear quality improvement. If Spec512
   (13.7 MB, 25.6M params at 4-bit) is indistinguishable from Shade (11.2 MB) in
   qualitative eval, there is no case for a separate Tier 3.
