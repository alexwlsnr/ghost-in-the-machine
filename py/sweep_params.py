#!/usr/bin/env python3
"""
Parameter sweep for BPE ternary models.
Tests temperature × rep_penalty × top_k combos and scores output quality.

Usage:
  .venv/bin/python3 py/sweep_params.py --checkpoint ckpt/shade_bpe_ternary.pt \
      --tokenizer data/bpe_4096.json
"""
import argparse, json, math, sys, torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bpe_tokenizer import BPETokenizer


# ── load model ────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    from train_transformer import TinyTransformer, TinyTransformerTernary
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a   = ck['architecture']
    arch_type = a.get('arch', a.get('type', 'classic'))
    cfg = {
        'vocab_size': a['vocab_size'],
        'd_model':    a['d_model'],
        'n_heads':    a['n_heads'],
        'n_layers':   a['n_layers'],
        'd_ff':       a['d_ff'],
        'max_len':    a['max_len'],
    }
    cls = TinyTransformerTernary if arch_type == 'ternary' else TinyTransformer
    model = cls(**cfg)
    state = {k.replace('_orig_mod.', ''): v for k, v in ck['model_state'].items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device), a


# ── generation ────────────────────────────────────────────────────────────────

def generate_bpe(model, tok: BPETokenizer, prompt: str, *,
                 max_new=80, temperature=0.7, top_k=0, top_p=0.0,
                 rep_penalty=1.0, preserve_case=False, device='cpu') -> str:

    prompt_str = prompt if preserve_case else prompt.upper()
    tokens = tok.encode(prompt_str)[:model.max_len - 2] + [tok.SEP]
    prompt_len = len(tokens)
    generated = []

    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1].clone()

            # Temperature
            next_logits = next_logits / max(temperature, 1e-8)

            # Repetition penalty — discount tokens already generated
            if rep_penalty != 1.0 and generated:
                for tid in set(generated):
                    if next_logits[tid] > 0:
                        next_logits[tid] /= rep_penalty
                    else:
                        next_logits[tid] *= rep_penalty

            # Top-k
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < topk_vals[-1]] = -float('inf')

            # Top-p (nucleus)
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # remove tokens with cumulative prob above threshold
                sorted_logits[cum_probs > top_p] = -float('inf')
                next_logits = torch.zeros_like(next_logits).scatter_(0, sorted_idx, sorted_logits)

            # Guard: if all logits are -inf (over-filtered), fall back to greedy
            if torch.all(next_logits == -float('inf')):
                next_token = torch.argmax(logits[0, -1]).item()
            else:
                probs = F.softmax(next_logits, dim=-1)
                probs = torch.clamp(probs, min=0)  # guard against fp noise
                next_token = torch.multinomial(probs, 1).item()

            if next_token in (tok.EOS, tok.PAD, tok.SEP):
                break

    return tok.decode(generated)


# ── scoring heuristics ────────────────────────────────────────────────────────

def score_response(prompt: str, response: str) -> dict:
    """Simple heuristic scoring — not ground truth, just a signal."""
    issues = []

    # Too short
    if len(response.strip()) < 5:
        issues.append('too_short')

    # Runon: response > 2× typical length
    if len(response) > 250:
        issues.append('runon')

    # Repetition: any 4-gram appearing 3+ times
    words = response.lower().split()
    if len(words) >= 8:
        fourgrams = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
        if any(fourgrams.count(g) >= 3 for g in set(fourgrams)):
            issues.append('repetition')

    # Off-topic drift: response contains prompt content verbatim (echoing)
    if len(prompt) > 4 and prompt.lower()[:6] in response.lower():
        issues.append('echo')

    score = max(0, 10 - len(issues) * 3)
    return {'score': score, 'issues': issues, 'len': len(response)}


# ── prompts ───────────────────────────────────────────────────────────────────

PROMPTS = [
    ("greeting",  "HELLO"),
    ("joke",      "TELL ME A JOKE"),
    ("factual",   "WHAT IS A BUS?"),
    ("howru",     "HOW ARE YOU?"),
    ("identity",  "TELL ME ABOUT YOURSELF"),
    ("multiturn", "HOW ARE YOU?|I AM GOOD, THANKS. WHAT IS YOUR NAME?"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer', default='data/bpe_4096.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='logs/param_sweep.json')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, args.device)
    tok = BPETokenizer(args.tokenizer)
    preserve_case = cfg.get('preserve_case', False)

    print(f"Arch: d={cfg['d_model']} L={cfg['n_layers']} vocab={cfg.get('vocab_size', '?')} preserve_case={preserve_case}\n")

    # ── parameter grid ────────────────────────────────────────────────────────
    temps       = [0.4, 0.5, 0.6, 0.7, 0.8]
    rep_pens    = [1.1, 1.2, 1.3, 1.4, 1.5]
    top_ks      = [0, 20, 40]        # 0 = disabled
    top_ps      = [0.0, 0.90, 0.95]  # 0 = disabled

    # Reduced grid: fix top_p=0 for top_k sweep, fix top_k=0 for top_p sweep
    combos = []
    for t in temps:
        for rp in rep_pens:
            for k in top_ks:
                combos.append(dict(temperature=t, rep_penalty=rp, top_k=k, top_p=0.0))
            for p in top_ps[1:]:  # skip 0.0 (same as top_k=0 already covered)
                combos.append(dict(temperature=t, rep_penalty=rp, top_k=0, top_p=p))

    print(f"Testing {len(combos)} parameter combinations × {len(PROMPTS)} prompts = {len(combos)*len(PROMPTS)} generations\n")

    results = []
    best_combos = {}  # prompt_key → best so far

    for i, params in enumerate(combos):
        combo_scores = []
        combo_outputs = {}
        for key, prompt in PROMPTS:
            response = generate_bpe(
                model, tok, prompt,
                max_new=100,
                preserve_case=preserve_case,
                device=args.device,
                **params
            )
            sc = score_response(prompt, response)
            combo_scores.append(sc['score'])
            combo_outputs[key] = response

        avg_score = sum(combo_scores) / len(combo_scores)
        entry = {**params, 'avg_score': avg_score, 'outputs': combo_outputs}
        results.append(entry)

        # Progress every 20 combos
        if (i + 1) % 20 == 0 or i == len(combos) - 1:
            print(f"  [{i+1:3d}/{len(combos)}] best so far: {max(r['avg_score'] for r in results):.1f}")

    # ── report top 10 ─────────────────────────────────────────────────────────
    results.sort(key=lambda r: r['avg_score'], reverse=True)

    print("\n" + "═"*72)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("═"*72)
    for rank, r in enumerate(results[:10], 1):
        p = r
        print(f"\n#{rank}  score={p['avg_score']:.1f}  "
              f"temp={p['temperature']}  rep={p['rep_penalty']}  "
              f"top_k={p['top_k']}  top_p={p['top_p']}")
        for key, prompt in PROMPTS[:4]:  # show first 4 prompts
            print(f"  {key:12s}: {r['outputs'][key][:100]}")

    # ── save full results ──────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {args.out}")

    # ── recommend ─────────────────────────────────────────────────────────────
    best = results[0]
    print(f"\n{'═'*72}")
    print("RECOMMENDED SETTINGS:")
    print(f"  temperature:  {best['temperature']}")
    print(f"  rep_penalty:  {best['rep_penalty']}")
    print(f"  top_k:        {best['top_k']}")
    print(f"  top_p:        {best['top_p']}")
    print("═"*72)


if __name__ == '__main__':
    main()
