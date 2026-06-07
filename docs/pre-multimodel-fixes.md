# Pre-Multi-Model Fixes

Code-level bugs and cleanup to land **before** the multi-model work in
[`multi-model-plan.md`](./multi-model-plan.md). These are the "immediate fixes"
split out from the plan review. Most are prerequisites: you cannot reliably
serialize, run, or evaluate a second model until #1–#4 are done.

Severity: 🔴 blocker · 🟠 important · 🟡 cleanup.

**Status** (branch `fix/pre-multimodel-fixes`): the hard gate (#1–#4) is cleared,
plus #6/#7. #5 is intentionally deferred to the data-pipeline rebuild; #8 is partly
done. Each fix below carries its resolution.

| # | Fix | Status |
|---|-----|--------|
| 1 | Canonical serializer (was missing) | ✅ `ec25fc4` — `py/serialize.py`, bit-exact; old serializers archived |
| 2 | TS arch-driven dims | ✅ `ec25fc4` — covered by `forward_arch.test.js` |
| 3 | Logits buffer alloc (`vocab*2`) | ✅ `56c242d` |
| 4 | Scratch margin sized from arch | ✅ `56c242d` |
| 5 | EOS / length filter | ⬜ deferred → multi-model Phase 2 (data pipeline) |
| 6 | Python `generate()` prompt chop | ✅ `41e8a55` |
| 7 | Parity harness + retire stale tests + CI | ✅ `6f01318`, `bf5c735` |
| 8 | Cleanup | 🟡 serializers consolidated; API-key path still open |

Regression net guarding all of this: **12 JS tests** (CI-gated via `node --test`) +
**3 Python tests** (`npm run test:py`, local — needs torch).

---

## ✅ 1. No canonical serializer — the shipped bundle had no source script

**Resolved — `ec25fc4`.** New `py/serialize.py` reproduces the shipped float32 bundle
bit-exact (token_embed × √d, pos_embed raw, head/biases/LN float32) and also emits a
4-bit build. The five old serializers + three `feedme*` variants are moved to
`py/archive/`. The `serialize_v3.py` `pos_embed × √d` bug is gone with it.

<details><summary>Original finding</summary>

`dist/transformer_model.bin` is **float32**, with `token_embed × √d` baked in,
`pos_embed` raw, all 69 manifest scales written as `1.0`. No tracked script produced
this format: `serialize_v3.py` packed to 4-bit, scaled `pos_embed` by √d wrongly
(`serialize_v3.py:81` vs `train_transformer.py:76-78` — a 16× error), and wrote to a
nonexistent `stage3/dist/`. Five divergent serializers in `py/` were how the bug happened.
</details>

## ✅ 2. Hardcoded dims in the TS orchestrator broke any non-Wisp model

**Resolved — `ec25fc4`.** The embedding views now use `arch.vocab_size` / `arch.max_len`
instead of the hardcoded `257` / `64`. `test/forward_arch.test.js` exercises a large
non-Wisp arch (1024 ctx) end-to-end to prove it loads and runs without OOB.

<details><summary>Original finding</summary>

`f32(S('token_embed'), 257 * d)` and `f32(S('pos_embed'), 64 * d)`: for Shade (ctx=128),
reading `peW` at position ≥ 64 ran past the typed-array view → NaN cascade → degenerate
sampling.
</details>

## ✅ 3. Latent buffer overrun on the logits (same family as commit ab59bc5)

**Resolved — `56c242d`.** Allocation corrected from `ba(arch.vocab_size + d)` to
`ba(arch.vocab_size * 2)` (zero-bias buffer + logits, `vocab_size` each). Harmless on
Wisp only because it was the terminal allocation and 16-byte rounding absorbed it —
fixed before a KV-cache buffer lands after it. Parity/generation goldens confirm no
behavior change.

## ✅ 4. Scratch margin was hardcoded and barely survived Specter

**Resolved — `56c242d`.** `loadModel`/`instantiateModel` now size headroom from the arch
via `forwardScratchBytes()` (mirrors the `ba(...)` allocations in `forward()`), replacing
the fixed 8 MB. A 1024-ctx arch needs ~13 MB and trapped with a `RangeError` under the
old margin — `forward_arch.test.js` was RED before, GREEN after. Wisp now uses ~0.65 MB.

## ⬜ 5. EOS is trained at the truncation point, not at sentence ends

**Deferred — multi-model Phase 2 (data-pipeline rebuild).** The fix is a data-generation
filter (`len(q) + len(r) + 1 ≤ ctx`, EOS only at genuine response ends), so it belongs
with `expand_data.py`, not pre-work — no retraining happens in pre-work.

<details><summary>Original finding</summary>

`make_sequence` (`train_transformer.py:193-199`) truncates over-length pairs and appends
EOS at the cut. 98.4% of the 2,046 pairs are ≥ 63 bytes (ctx 64) — nearly every example
teaches "emit EOS mid-sentence at byte 63," which is why EOS is imprecise and the UI needs
the sentence-trim hack (`dist/index.html:354-369`).
</details>

## ✅ 6. Python `generate()` silently chopped the prompt

**Resolved — `41e8a55`.** Prompt is now capped at the context window (`max_len - 1`,
mirroring TS) and `prompt_len` is taken from the kept tokens, so the returned slice no
longer drops the first generated token. `test/test_generate.py` covers full-prompt-fed,
first-token-not-dropped, and window-cap with a deterministic stub model.

<details><summary>Original finding</summary>

`tokens = tokens[:model.max_len - max_new]` cut "HELLO" → "HELL" at the defaults, and
`prompt_len = len(encode(prompt.upper()))` then dropped the first generated token in the
returned slice.
</details>

## ✅ 7. Stale tests; no transformer parity test existed

**Resolved — `6f01318` (tests) + `bf5c735` (CI).** Added `kernel.test.js` (7), `parity.test.js`
(2, vs committed PyTorch reference), `generation.test.js` (2, seeded-RNG determinism +
golden), and `forward_arch.test.js` (1). Stale `stage2_*.js` removed. `node --test` runs
in CI before deploy. Each net was proven to catch a regression via mutation (dropped
matmul bias; injected `pos_embed × √d`).

## 🟡 8. Cleanup

- ✅ Serializers consolidated — five variants + `feedme*` archived to `py/archive/` (`ec25fc4`).
- ⬜ `py/batch_distill.py:10` still hardcodes an API-key file path under `~/Downloads` — move
  to an env var / CLI arg.
- ⬜ `batch_distill.clean()` strips to a 39-char charset (no `.,'`) — reconcile when
  consolidating the distill scripts (multi-model Phase 2).
- ⬜ README and boot banner hardcode "256d × 4L | 13MB" — addressed by the `models.json`
  registry work in the multi-model plan.

---

## Order (as executed)

1. ✅ **#1 canonical serializer + #7 parity harness** — reproducibility floor.
2. ✅ **#2 + #3 + #4 TS arch-driven** — de-hardcoded views, fixed logits alloc, arch-sized
   scratch; validated on a large synthetic arch.
3. ⬜ **#5 EOS / length filter** — folded into the data-pipeline rebuild (Phase 2).
4. ✅ **#6 Python generate()** — done. 🟡 **#8 cleanup** — partial (API-key path open).
