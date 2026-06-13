# Chat eval harness

Pairwise LLM-judge eval for the micro-LLM zoo. Generates responses from two
models on a fixed prompt set, then asks a neutral cloud judge which better fits
the terse persona + answers well. Reports a candidate win rate with per-category
and per-dimension breakdowns and a position-bias check.

Built as a lean v1 of the plan in `../M3_Eval_Plan.md`.

## Usage

```bash
# 1. Generate responses (PyTorch — fast, faithful; use for any .pt checkpoint)
.venv/bin/python3 eval/generate_pt.py <ckpt.pt> <tokenizer.json> <tag> \
    --device cuda --set eval_set_100.jsonl
#    legacy fallback for served-only models (.bin/.json, no .pt):
#    EVAL_SET=eval_set_100.jsonl node eval/generate.mjs <model_prefix> <tag>

# 2. Judge candidate vs baseline
#    cloud (default): neutral DeepSeek arbiter, needs OPENCODE_API_KEY
export OPENCODE_API_KEY=sk-...                 # opencode-go bearer token
python3 eval/judge.py <baseline_tag> <candidate_tag> --judge deepseek --workers 8

#    local cross-check: any OpenAI-compatible llama-server
llama-server -m <judge.gguf> --port 8091 -ngl 99 -c 4096 --parallel 2 &
JUDGE_MODEL=<gguf-id> python3 eval/judge.py <baseline_tag> <candidate_tag> \
    --judge local --workers 2
```

### Judges

- `--judge deepseek` — opencode-go `deepseek-v4-flash`, neutral family, 8 workers.
- `--judge local` — local llama-server via `JUDGE_URL` (default `:8091`) +
  `JUDGE_MODEL`. Use a NON-teacher family (e.g. Nemotron-4B, **not** Gemma — the
  distillation teacher — to avoid self-preference bias). 2 workers (VRAM-bound).

Cross-judge agreement is the robustness check: DeepSeek and Nemotron-4B both
scored Spectre ep60 vs ep36 a wash (54% / 56%), and the reasoning-disabled
teacher (Gemma-E4B-distill) agreed at 46% — three independent judges, all inside
the noise floor, so the verdict isn't a single-judge artifact (and the teacher
showed no self-preference blow-up).

## Files

- `eval_set.jsonl` / `eval_set_100.jsonl` — prompt sets (id, category, prompt)
- `generate_pt.py` — PyTorch generator (source of truth; GPU/CPU, batched)
- `generate.mjs` — node/WASM generator (legacy fallback for served `.bin/.json`)
- `judge.py` — pairwise judge via opencode-go `deepseek-v4-flash`
- `out_<tag>.jsonl` — generated responses (gitignored)

## Judge settings (validated)

`max_tokens` is per-backend (deepseek 4000, local 512):
- cloud deepseek `reasoning_effort: low` + `max_tokens: 4000` — 0/100 truncation
  failures (at 2000, ~10% truncated mid-reasoning → empty content).
- local `max_tokens: 512` — enough for the JSON+reason from a non-reasoning model.
- `--workers`: deepseek 8 (~2.1 s/prompt, ~3.5 min/100); local 2–4 (VRAM-bound).
- Position is randomised per prompt (seeded) and a slot-bias check is reported.
- Empty `content` falls back to the JSON restated at the end of `reasoning_content`.

## Local-judge gotchas (llama.cpp / llama-swap)

Learned the hard way running Gemma-E4B as a local judge:

- **Warm the model first.** llama-swap loads on demand; if the run's opening
  concurrent burst arrives mid-load, those calls get HTTP 400. Send one priming
  request before the eval.
- **Per-slot context = `--ctx-size / -np`.** A throughput profile like
  `-np 32 --ctx-size 1024` gives 32 tokens/slot — far too small for a ~300-token
  judge prompt (→ 400s on long prompts, truncated JSON on short ones). For a
  judge, run a profile with few slots and large context (e.g. `-np 4 --ctx-size
  8192`). The Gemma *distill* profiles are tuned for single-token scoring, not
  judging — launch a dedicated `llama-server ... --reasoning off --ctx-size 8192`
  instead of reusing them.
- A reasoning-disabled small model (Gemma-E4B-distill, reasoning off) is the
  *fastest* judge here: ~0.3 s/prompt, ~30 s for 100, since it emits only the
  short JSON (no reasoning tokens).

## Notes

- Noise floor ~±10pp at n=100 — treat win rates <58% as a wash. Not for
  splitting near-identical checkpoints.
- Cross-judge robustness: DeepSeek (cloud), Nemotron-4B (local), and the
  Gemma-E4B teacher are three distinct families; agreement across them means a
  verdict isn't a single-judge quirk. Use a NON-teacher family as the primary
  arbiter to keep self-preference bias low.
