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
scored Spectre ep60 vs ep36 a wash (54% / 56%), confirming the verdict isn't a
single-judge artifact.

## Files

- `eval_set.jsonl` / `eval_set_100.jsonl` — prompt sets (id, category, prompt)
- `generate_pt.py` — PyTorch generator (source of truth; GPU/CPU, batched)
- `generate.mjs` — node/WASM generator (legacy fallback for served `.bin/.json`)
- `judge.py` — pairwise judge via opencode-go `deepseek-v4-flash`
- `out_<tag>.jsonl` — generated responses (gitignored)

## Judge settings (validated)

- `reasoning_effort: low`, `max_tokens: 4000` — 0/100 truncation failures.
  (At 2000, ~10% of calls truncated mid-reasoning and returned empty content.)
- `--workers 8` — ~2.1 s/prompt wall, ~3.5 min for 100 prompts. Go endpoint
  tolerates 8 concurrent without rate-limiting.
- Position is randomised per prompt (seeded) and a slot-bias check is reported.
- Empty `content` falls back to the JSON restated at the end of `reasoning_content`.

## Notes

- Noise floor ~±10pp at n=100 — treat win rates <58% as a wash. Not for
  splitting near-identical checkpoints.
- The judge (DeepSeek) is a different family from the distillation teacher
  (Gemma) and the models (ternary) — keeps self-preference bias low.
