# Training process — dataset → train → serialize → eval

The canonical end-to-end pipeline for a new model (e.g. Spectre v3). Follow it in
order so runs are comparable. Every stage names the real script. The goal,
per memory, is **quality-per-byte: match Llama-3.2-1B-ghost persona quality under
100MB** — so every run ends by being judged against that ceiling *and* the
current best (regression check).

All commands run from the repo root with the venv python (`.venv/bin/python3`).

---

## 0. Decide the run's variables up front

Write these down in the training script header (copy an existing
`scripts/train_*.sh`) so the run is reproducible:

- **arch**: `ternary_modern` (RoPE + RMSNorm + SwiGLU + ternary; the deployable one)
- **size**: `--d-model --n-layers --n-heads --d-ff` (+ `--n-kv-heads` for GQA)
- **ctx**: `--max-len` (512 for Spectre tier)
- **data blend** (stage 1) and **tokenizer**
- **target serialized size** — keep an eye on the 100MB budget

---

## 1. Build the dataset

`py/build_revenant_dataset.py` assembles a blend into a pipe-delimited pairs file
(`Q|R` single-turn, `Q1|R1|Q2|R2|…` multi-turn).

```bash
.venv/bin/python3 py/build_revenant_dataset.py --blend spectre_v2 --out data/<name>_train.txt
# blends: baseline | factual | quality | scale | domain | memory | revenant | spectre_v2
```

- **Loud-fail on missing sources.** The builder aborts with a banner if any HF
  source fails (so the blend is never silently short). Only pass
  `--allow-failed-sources` if you've consciously accepted the gap.
- **For v3, apply the lessons** (see `[[spectre-v3-lessons]]` memory):
  - drop SmolTalk tool-call / code subsets (they teach `<tool_call>` noise),
  - **scrub assistant-mode / AI-disclaimer boilerplate** ("As an AI…", "I am a
    language model", "I can provide information and assistance") — this is what
    cost ep60 the persona head-to-head vs Llama-1B,
  - add explicit **multi-turn fact-retention** examples (state fact → later ask)
    — multi-turn is the one category v2 never beat v1 on.
- **Bonemaxxing (signature flavor):** `py/build_bones.py` expands `bones.json`
  (187 punchlines) → `data/bones_train.txt` (747 pairs: joke-request→joke,
  setup→punchline, two-turn delivery). The bone joke is a kept signature — `cat`
  this into the blend, but dose it lightly (1× is plenty; tiny models
  mode-collapse onto bone jokes if over-weighted).
- **Optional persona/quality filter:** `py/score_dataset.py` (llama-server judge,
  rates terseness+comprehension). Currently PAUSED — small judges were unreliable;
  needs grammar-constrained output + a stronger judge before trusting it.

## 2. Tokenizer

Use the **byte-level BPE** (llama.cpp-compatible, the adopted standard):

```bash
.venv/bin/python3 py/train_bpe_bytelevel.py --input data/<name>_train.txt --out data/bpe_bytelevel_4099.json
# (reuse the existing data/bpe_bytelevel_4099.json unless vocab/domain changed)
```

Specials: `<PAD>=0 <EOS>=1 <SEP>=2`. The trainer auto-detects HF-format tokenizers
(`byte_bpe_tokenizer.is_hf_tokenizer`) vs legacy char-BPE. Prefer byte-level for
all new runs — it's GGUF/llama.cpp interoperable and keeps embeddings smaller.

## 3. Train

`py/train_transformer.py`. Copy a `scripts/train_*.sh` (e.g. `train_spectre_v2.sh`
or `train_wisp_gqa.sh`) and edit the header. Canonical invocation:

```bash
nohup .venv/bin/python3 -u py/train_transformer.py \
  --file data/<name>_train.txt \
  --tokenizer data/bpe_bytelevel_4099.json \
  --checkpoint ckpt/<name>.pt \
  --arch ternary_modern \
  --d-model 512 --n-heads 8 --n-layers 8 --d-ff 2048 \
  --max-len 512 --batch-size 32 --lr 0.0006 \
  --epochs 100 --val-frac 0.02 --patience 25 \
  --truncate --amp --mask-query-loss \
  --status-file logs/<name>_status.json \
  --device cuda > logs/<name>_train.log 2>&1 &
```

- `--n-kv-heads N` enables GQA (N < n_heads). NB: the A/B showed GQA is **not
  free** (~0.15 val-loss cost at 2× KV reduction) — only use if KV-cache size is
  the binding constraint.
- **Status JSON** (`--status-file`) is how we monitor: epoch, train/val loss,
  best_val_loss + best_epoch, eta_seconds, state, pid. The hourly check reads it.
- **Second box (cuboid):** for parallel runs, ship to `alex-cuboid` (RTX 3070 8GB,
  repo `~/dev/ghost-in-the-machine`, no tmux → use `nohup`). See `[[cuboid]]`.
- Resume with `--resume ckpt/<name>.pt` (same path = continue; different = finetune).

## 4. Serialize to the deployable format

`py/serialize.py` → `dist/model_<name>.{bin,json}` (ternary-packed, head-tied,
tokenizer embedded).

```bash
.venv/bin/python3 py/serialize.py ckpt/<name>.pt --out dist/model_<name>
# ternary_modern serializes ternary by default; head is tied (token_embed reused) → ~8MB saved
```

- **Check the size** against the 100MB budget. Embeddings are the uncompressed
  part (`[[embedding-size-frontier]]`) — quantize embeddings before growing vocab.
- **`dist/*.js` drift hazard** (`[[dist-ts-drift-hazard]]`): `dist/tier2_transformer.js`
  has hand-edits; if you recompile the TS, diff first so you don't revert fixes.
- **Register in the UI**: add an entry to `dist/models.json` (id, bin, params, ctx,
  temp, top_k, top_p, max_new) — the eval generator reads sampling params from here.
- **Optional GGUF** (for llama.cpp): `py/convert_to_gguf.py --checkpoint ckpt/<name>.pt
  --tokenizer data/bpe_bytelevel_4099.json --out dist/gguf/<name>.gguf` (F32 norms,
  folded embed scale — see `[[gguf-interop-findings]]`).

## 5. Eval — the gate

Pairwise LLM-judge harness in `eval/`. See `eval/README.md`. **Every run is judged
against two opponents minimum:**

1. **Llama-3.2-1B-ghost** — the *ceiling/target*. How close are we to a prompted
   1B in <100MB? (Baseline gap as of 2026-06-13: Llama-ghost beats ep60 **94%**.)
2. **Current best deployed model** — the *regression check*. Did we actually
   improve, or just move sideways? (e.g. vs `spectre-v2-ep60`.)

```bash
# A) generate the new model's responses via the DEPLOYED engine (fast + faithful)
EVAL_SET=eval_set_200.jsonl node eval/gen_engine.mjs model_<name> <name> 1234

# B) generate / reuse the two opponents
#    current best (served snapshot) — engine runner:
EVAL_SET=eval_set_200.jsonl node eval/gen_engine.mjs model_spectre_v2_ep60 ep60 1234
#    Llama-1B-ghost — llama-server + ghost system prompt:
llama-server -hf bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0 --port 8094 -ngl 99 -c 8192 -np 4 &
python3 eval/gen_prompted.py llama1b_ghost --url http://127.0.0.1:8094/v1/chat/completions \
  --model llama --system eval/ghost_system.txt --set eval_set_200.jsonl

# C) judge (needs OPENCODE_API_KEY for the cloud arbiter)
export OPENCODE_API_KEY=...                       # opencode-go bearer token
python3 eval/judge.py ep60 <name>          --judge deepseek --workers 8   # regression
python3 eval/judge.py <name> llama1b_ghost --judge deepseek --workers 8   # distance to target
```

- **`.pt` available?** Use `eval/generate_pt.py` instead of `gen_engine.mjs`
  (faster on GPU, also faithful). Served-only snapshots (no `.pt`) must use the
  engine runner.
- **Read it right** (see `[[chat-eval-harness]]`):
  - noise floor ±10pp at n=100 → treat <58% as a wash; 200-set gives reliable
    per-category reads (n=40 each: persona/terse/general/edge/multiturn).
  - the judge is **factuality-blind** — it scores persona/coherence, not
    correctness. Don't read terse-category wins as "knows more facts".
  - **multi-turn** is the category to watch for v3 (v2's weak spot).
  - cross-check with `--judge local` (a non-teacher family) when a verdict matters.

## 6. Decide & promote

- **Ship** if the new model beats current-best clearly (>58%) on the regression
  judge *and* doesn't regress multi-turn.
- Record the distance-to-target (win rate vs Llama-1B-ghost) — that's the
  project's actual scoreboard.
- Promote: update `dist/models.json` default, commit, note the eval numbers in the
  commit message.

---

## One-line summary

`build_revenant_dataset → train_bpe_bytelevel → train_transformer → serialize
→ eval (vs current-best for regression, vs Llama-1B-ghost for distance-to-target)`.
