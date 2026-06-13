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

# 2. Judge candidate vs baseline (needs OPENCODE_API_KEY)
export OPENCODE_API_KEY=sk-...                 # opencode-go bearer token
python3 eval/judge.py <baseline_tag> <candidate_tag> --workers 8
```

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
