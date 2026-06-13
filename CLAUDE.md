# Project notes for Claude

## Infrastructure

- **Cuboid** (secondary training box): I keep its details in persistent memory — SSH host
  `alex-cuboid`, RTX 3070 8GB, repo at `~/dev/ghost-in-the-machine` (not the micro-llm path),
  venv specifics, and how to install packages / ship code there. See the `cuboid` memory rather
  than re-discovering it each session.

## Standard training process

**`TRAINING_PROCESS.md` is the canonical pipeline** — follow it for every new model so
runs stay comparable: dataset build (`build_revenant_dataset.py`) → byte-level tokenizer
(`train_bpe_bytelevel.py`) → train (`train_transformer.py`, `ternary_modern`) → serialize
(`serialize.py` → `dist/`) → **eval gate** (`eval/`).

Standing conventions:

- **Eval every run against TWO opponents** (`eval/`, pairwise LLM-judge): the **current
  best deployed model** (regression check — ship only if >58%) and **Llama-3.2-1B-ghost**
  (distance-to-target). Generate via `eval/gen_engine.mjs` (runs the deployed engine
  headless — fast + bug-for-bug faithful) for served models, or `generate_pt.py` for `.pt`.
  Judge with `eval/judge.py --judge deepseek` (set `OPENCODE_API_KEY`). The judge is
  **factuality-blind** (scores persona/coherence, not correctness); ±10pp noise floor at
  n=100. Watch the **multi-turn** category — v2's weak spot.
- **The project's north star: match Llama-3.2-1B-ghost persona quality in <100MB**
  (quality-per-byte). Persona is easy to prompt into a 1B model; *smallness is the moat*.
  Current baseline gap: Llama-1B-ghost beats our 35M ep60 **94%**.
- **Data cleaning for v3** (two levers): cheap **regex scrub** of tool-call / AI-disclaimer
  boilerplate (the assistant-leakage that cost ep60 the persona), plus the **persona
  scorer** (`score_dataset.py`, now working — GBNF grammar + reasoning-off judge) for the
  subtler off-persona long tail. Also add **multi-turn fact-retention** examples.
- **Bone jokes are a KEPT signature** — do not scrub them. `py/build_bones.py` →
  `data/bones_train.txt` deliberately reinforces them (dose lightly; tiny models
  mode-collapse onto bone jokes if over-weighted).
- Deeper rationale for all of the above lives in persistent memory (e.g.
  `project-target-llama1b-under-100mb`, `chat-eval-harness`, `spectre-v3-lessons`,
  `persona-scorer-unblocked`).
