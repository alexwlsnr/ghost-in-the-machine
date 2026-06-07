# Pre-Multi-Model Fixes

Code-level bugs and cleanup to land **before** the multi-model work in
[`multi-model-plan.md`](./multi-model-plan.md). These are the "immediate fixes"
split out from the plan review. Most are prerequisites: you cannot reliably
serialize, run, or evaluate a second model until #1–#4 are done.

Severity: 🔴 blocker · 🟠 important · 🟡 cleanup.

---

## 🔴 1. No canonical serializer — the shipped bundle has no source script

`dist/transformer_model.bin` is **float32**, with `token_embed × √d` baked in,
`pos_embed` raw, all 69 manifest scales written as `1.0`. Verified against
`transformer_model_eos.pt`. **No tracked script produces this format:**

- `py/serialize_v3.py` (README points here) packs weights to **4-bit**, which the
  current float32 TS path can't read.
- It also has a real bug: it scales **`pos_embed` by √d** (`serialize_v3.py:81`),
  but training only scales token embeddings (`train_transformer.py:76-78`) — a 16×
  error on positional embeddings.
- It writes to `stage3/dist/`, a path that doesn't exist here.

There are **five** serializer variants in `py/` (`serialize_transformer.py`,
`serialize_transformer_fixed.py`, `serialize_transformer_v2.py`, `serialize_v3.py`,
`tier2_serialization.py`) plus three `feedme*` variants — this divergence is how the
bug happened.

**Fix:** Write one canonical serializer that reproduces the *current shipped float32
format exactly* (token_embed × √d, pos_embed raw, head raw, biases/LN float32),
verified bit-exact against the live bundle. Delete or archive the other four.
This is the foundation for everything in the multi-model plan.

## 🔴 2. Hardcoded dims in the TS orchestrator break any non-Wisp model

`ts/src/tier2_transformer.ts:92-93`:
```ts
const teW = f32(S('token_embed'), 257 * d);   // vocab is 258 → must be arch.vocab_size
const peW = f32(S('pos_embed'), 64 * d);       // must be arch.max_len
```
For Shade (ctx=128), reading `peW` at position ≥ 64 runs past the typed-array view →
`undefined` → NaN cascade through every layer → sampling degenerates. These two lines
(plus the scratch margin, #4) are the bulk of "TS orchestrator: arch-driven."

**Fix:** use `arch.vocab_size` and `arch.max_len` for the view lengths.

## 🟠 3. Latent buffer overrun on the logits (same family as commit ab59bc5)

`ts/src/tier2_transformer.ts:89` allocates `ba(arch.vocab_size + d)` but uses `vocab`
(zero-bias buffer) + `vocab` (logits). When `vocab > d` — true for Wisp, 258 > 256 —
the last 2 floats of the logits (the **PAD and EOS logits**, exactly what ab59bc5
fixed) are written past the allocation. It works today only because `ba`'s 16-byte
alignment rounding happens to absorb 8 bytes.

**Fix:** allocate `ba(arch.vocab_size * 2)`.

## 🟠 4. Scratch margin is hardcoded and barely survives Specter

`ts/src/tier2_transformer.ts:47` adds a fixed `8 * 1024 * 1024` of headroom. Measured
forward-pass scratch for Specter (seq 256, d 768, d_ff 3072) is **~7.0 MB** — it
squeaks under, until a KV cache or anything else is added, then it corrupts/traps.

**Fix:** compute the margin from `arch` (seq, d, d_ff, vocab) instead of hardcoding.

## 🟠 5. EOS is trained at the truncation point, not at sentence ends

`train_transformer.py:193-199`: `make_sequence` truncates over-length pairs and appends
EOS *at the cut*. Measured: **98.4% of the 2,046 training pairs are ≥ 63 bytes** (mean
117, ctx 64) — so nearly every example teaches "emit EOS mid-sentence at byte 63."
This is why EOS prediction is imprecise and the UI needs the sentence-trim hack
(`dist/index.html:354-369`).

**Fix:** filter to `len(q) + len(r) + 1 ≤ ctx` and only append EOS at a genuine
response end (overlaps with the data-format work in the multi-model plan; the filter
alone fixes the existing data and likely lets the UI trim hack be deleted).

## 🟡 6. Python `generate()` silently chops the prompt

`train_transformer.py:336`: `tokens = tokens[:model.max_len - max_new]` — with
`max_new=60` and ctx 64, the prompt is cut to **4 bytes** ("HELLO" → "HELL"). The TS
version handles this correctly (cap prompt at the window, stop when full).

**Fix:** mirror the TS approach. Matters because you'll use the Python side to
eyeball Shade/Specter quality during training.

## 🟡 7. Stale tests; no transformer parity test exists

`test/stage2_test.js` / `stage2_integration_test.js` load vectors from `test/stage2/`
(not in the repo) and target the **old z80 MLP kernel**, not the transformer. There is
no parity harness for the transformer at all.

**Fix:** add the Python-vs-TS/Wasm parity harness (also listed as Phase 1 of the
multi-model plan); retire or rewrite the stage2 tests.

## 🟡 8. Cleanup

- `py/batch_distill.py:10` hardcodes an API-key file path under `~/Downloads` — move to
  an env var / CLI arg.
- `batch_distill.clean()` strips to a 39-char charset (no `.,'`) — inconsistent with the
  transformer data; reconcile when consolidating the distill scripts.
- Consolidate the five serializer variants (see #1) and three `feedme*` variants down to
  one each.
- README and boot banner hardcode "256d × 4L | 13MB" — will be wrong the moment a second
  model exists (registry work is in the multi-model plan).

---

## Suggested order

1. **#1 canonical serializer + #7 parity harness** — reproducibility floor; everything
   downstream depends on it.
2. **#2 + #3 + #4 TS arch-driven** — de-hardcode views, fix logits alloc, compute scratch
   margin. Sanity-check by serializing + running a throwaway Shade-config model.
3. **#5 EOS / length filter** — fold into the data-pipeline rebuild.
4. **#6 Python generate() + #8 cleanup** — low-risk, do alongside.

#1–#4 are the hard gate: until they're done, the multi-model plan's "train Shade" step
produces a model that can't be serialized or run correctly.
