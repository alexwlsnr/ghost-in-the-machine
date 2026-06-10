#!/usr/bin/env python3
"""
Verbose parameter sweep — logs every response for subjective review.

Usage:
  .venv/bin/python3 py/sweep_verbose.py --checkpoint ckpt/shade_bpe_ternary.pt \
      --tokenizer data/bpe_4096.json
"""
import argparse, json, sys, torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bpe_tokenizer import BPETokenizer


# ── load model ────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    from train_transformer import TinyTransformer, TinyTransformerTernary
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck['architecture']
    arch_type = a.get('arch', a.get('type', 'classic'))
    cfg = {
        'vocab_size': a['vocab_size'], 'd_model': a['d_model'],
        'n_heads':    a['n_heads'],    'n_layers': a['n_layers'],
        'd_ff':       a['d_ff'],       'max_len':  a['max_len'],
    }
    cls = TinyTransformerTernary if arch_type == 'ternary' else TinyTransformer
    model = cls(**cfg)
    state = {k.replace('_orig_mod.', ''): v for k, v in ck['model_state'].items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device), a


# ── generation ────────────────────────────────────────────────────────────────

def generate_bpe(model, tok, prompt, *, max_new=100, temperature=0.7,
                 top_k=0, top_p=0.0, rep_penalty=1.0,
                 preserve_case=False, device='cpu') -> str:
    prompt_str = prompt if preserve_case else prompt.upper()
    tokens = tok.encode(prompt_str)[:model.max_len - 2] + [tok.SEP]
    generated = []

    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1].clone()

            next_logits = next_logits / max(temperature, 1e-8)

            if rep_penalty != 1.0 and generated:
                for tid in set(generated):
                    if next_logits[tid] > 0:
                        next_logits[tid] /= rep_penalty
                    else:
                        next_logits[tid] *= rep_penalty

            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < topk_vals[-1]] = -float('inf')

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cum_probs > top_p] = -float('inf')
                next_logits = torch.zeros_like(next_logits).scatter_(0, sorted_idx, sorted_logits)

            if torch.all(next_logits == -float('inf')):
                next_token = torch.argmax(logits[0, -1]).item()
            else:
                probs = F.softmax(next_logits, dim=-1)
                probs = torch.clamp(probs, min=0)
                next_token = torch.multinomial(probs, 1).item()

            if next_token in (tok.EOS, tok.PAD):
                break
            if next_token != tok.SEP:
                tokens.append(next_token)
                generated.append(next_token)
            if len(tokens) >= model.max_len:
                break

    return tok.decode(generated)


# ── scoring heuristics ────────────────────────────────────────────────────────

def score(prompt: str, response: str) -> dict:
    issues = []
    if len(response.strip()) < 5:
        issues.append('too_short')
    if len(response) > 250:
        issues.append('runon')
    words = response.lower().split()
    if len(words) >= 8:
        fourgrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
        if any(fourgrams.count(g) >= 3 for g in set(fourgrams)):
            issues.append('repetition')
    if len(prompt) > 4 and prompt.lower()[:6] in response.lower():
        issues.append('echo')
    return {'score': max(0, 10 - len(issues) * 3), 'issues': issues, 'len': len(response)}


# ── prompts ───────────────────────────────────────────────────────────────────
# Varied across categories: greeting, small talk, factual, emotional,
# opinion, clarification, multi-turn, ambiguous, meta, off-the-wall

PROMPTS = [
    # Greetings / social
    ("greeting_hi",       "HELLO"),
    ("greeting_hey",      "HEY"),
    ("howru",             "HOW ARE YOU?"),
    ("small_talk",        "WHAT ARE YOU UP TO?"),

    # Factual / definitional
    ("what_is_bus",       "WHAT IS A BUS?"),
    ("what_is_rain",      "WHY DOES IT RAIN?"),
    ("capital_france",    "WHAT IS THE CAPITAL OF FRANCE?"),

    # Emotional / supportive
    ("feeling_sad",       "I'M FEELING REALLY SAD TODAY"),
    ("good_news",         "I GOT A PROMOTION!"),
    ("im_bored",          "I'M BORED"),

    # Opinion / creative
    ("joke",              "TELL ME A JOKE"),
    ("fav_colour",        "WHAT IS YOUR FAVOURITE COLOUR?"),
    ("recommend",         "WHAT SHOULD I DO THIS WEEKEND?"),

    # Identity / meta
    ("who_are_you",       "WHO ARE YOU?"),
    ("can_you_help",      "CAN YOU HELP ME?"),

    # Multi-turn (SEP-separated)
    ("multiturn_name",    "HOW ARE YOU?|I AM GOOD THANKS. WHAT IS YOUR NAME?"),
    ("multiturn_topic",   "TELL ME A JOKE|THAT WAS FUNNY! TELL ME ANOTHER"),
    ("multiturn_help",    "I NEED HELP|WHAT KIND OF HELP DO YOU NEED?|I NEED HELP WITH MY COMPUTER"),

    # Ambiguous / tricky
    ("one_word",          "OK"),
    ("just_thanks",       "THANKS"),
]

CATEGORIES = {
    "social":    ["greeting_hi", "greeting_hey", "howru", "small_talk"],
    "factual":   ["what_is_bus", "what_is_rain", "capital_france"],
    "emotional": ["feeling_sad", "good_news", "im_bored"],
    "creative":  ["joke", "fav_colour", "recommend"],
    "identity":  ["who_are_you", "can_you_help"],
    "multiturn": ["multiturn_name", "multiturn_topic", "multiturn_help"],
    "tricky":    ["one_word", "just_thanks"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer', default='data/bpe_4096.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='logs/sweep_verbose.json')
    parser.add_argument('--log', default='logs/sweep_verbose.md')
    # Optionally run a subset of combos for quick inspection
    parser.add_argument('--focused', action='store_true',
                        help='Only run a focused set of promising combos instead of full grid')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading model from {args.checkpoint}...", flush=True)
    model, cfg = load_model(args.checkpoint, args.device)
    tok = BPETokenizer(args.tokenizer)
    preserve_case = cfg.get('preserve_case', False)

    print(f"Arch: d={cfg['d_model']} L={cfg['n_layers']} vocab={cfg.get('vocab_size')} "
          f"preserve_case={preserve_case}\n", flush=True)

    # ── parameter grid ────────────────────────────────────────────────────────
    if args.focused:
        # Focused: sweep temps and rep penalties that looked best in the heuristic run,
        # with top_k/top_p variations most likely to fix runon
        combos = [
            dict(temperature=t, rep_penalty=rp, top_k=k, top_p=p)
            for t   in [0.5, 0.6, 0.7]
            for rp  in [1.2, 1.3, 1.4, 1.5]
            for k, p in [(0, 0.0), (20, 0.0), (40, 0.0), (0, 0.90), (0, 0.95)]
        ]
    else:
        # Full grid
        combos = []
        for t in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for rp in [1.1, 1.2, 1.3, 1.4, 1.5]:
                for k in [0, 20, 40]:
                    combos.append(dict(temperature=t, rep_penalty=rp, top_k=k, top_p=0.0))
                for p in [0.90, 0.95]:
                    combos.append(dict(temperature=t, rep_penalty=rp, top_k=0, top_p=p))

    prompt_map = dict(PROMPTS)
    print(f"Testing {len(combos)} combos × {len(PROMPTS)} prompts = "
          f"{len(combos)*len(PROMPTS)} generations\n", flush=True)

    results = []
    for i, params in enumerate(combos):
        outputs = {}
        scores  = {}
        for key, prompt in PROMPTS:
            resp = generate_bpe(
                model, tok, prompt,
                max_new=100,
                preserve_case=preserve_case,
                device=args.device,
                **params,
            )
            sc = score(prompt, resp)
            outputs[key] = resp
            scores[key]  = sc

        avg = sum(s['score'] for s in scores.values()) / len(scores)
        all_issues = [i for s in scores.values() for i in s['issues']]
        results.append({**params, 'avg_score': avg, 'outputs': outputs,
                        'scores': scores, 'issue_counts': {
                            iss: all_issues.count(iss) for iss in set(all_issues)
                        }})

        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            print(f"  [{i+1:3d}/{len(combos)}] avg_best={max(r['avg_score'] for r in results):.1f}",
                  flush=True)

    results.sort(key=lambda r: r['avg_score'], reverse=True)

    # ── save JSON ─────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nJSON saved to {args.out}", flush=True)

    # ── write markdown report ─────────────────────────────────────────────────
    lines = []
    lines.append("# Shade BPE Ternary — Verbose Parameter Sweep\n")
    lines.append(f"Model: `{args.checkpoint}` | d={cfg['d_model']} L={cfg['n_layers']} "
                 f"preserve_case={preserve_case}\n")
    lines.append(f"{len(combos)} combos × {len(PROMPTS)} prompts\n")
    lines.append("---\n")

    for rank, r in enumerate(results, 1):
        p = r
        issue_str = ', '.join(f"{k}×{v}" for k, v in r['issue_counts'].items()) or 'none'
        lines.append(f"\n## #{rank}  score={p['avg_score']:.1f}  "
                     f"temp={p['temperature']}  rep={p['rep_penalty']}  "
                     f"top_k={p['top_k']}  top_p={p['top_p']}\n")
        lines.append(f"*Issues: {issue_str}*\n")

        # Group by category for readability
        for cat, keys in CATEGORIES.items():
            lines.append(f"\n**{cat}**\n")
            for key in keys:
                resp   = r['outputs'].get(key, '')
                sc     = r['scores'].get(key, {})
                prompt = prompt_map.get(key, key)
                flag   = ' ⚠' if sc.get('issues') else ''
                lines.append(f"- `{prompt}` → {resp}{flag}\n")

        lines.append("\n---\n")

    Path(args.log).write_text(''.join(lines))
    print(f"Markdown report saved to {args.log}", flush=True)

    # ── console summary: top 5 with all outputs ───────────────────────────────
    print("\n" + "═"*72)
    print("TOP 5 — FULL OUTPUT")
    print("═"*72)
    for rank, r in enumerate(results[:5], 1):
        issue_str = ', '.join(f"{k}×{v}" for k, v in r['issue_counts'].items()) or 'none'
        print(f"\n{'─'*72}")
        print(f"#{rank}  score={r['avg_score']:.1f}  temp={r['temperature']}  "
              f"rep={r['rep_penalty']}  top_k={r['top_k']}  top_p={r['top_p']}")
        print(f"issues: {issue_str}")
        for key, prompt in PROMPTS:
            resp = r['outputs'][key]
            flag = ' ⚠' if r['scores'][key]['issues'] else ''
            print(f"  {key:18s}  [{prompt[:30]:30s}]  →  {resp[:120]}{flag}")

    best = results[0]
    print(f"\n{'═'*72}")
    print("RECOMMENDED:")
    print(f"  temperature: {best['temperature']}")
    print(f"  rep_penalty: {best['rep_penalty']}")
    print(f"  top_k:       {best['top_k']}")
    print(f"  top_p:       {best['top_p']}")
    print("═"*72)


if __name__ == '__main__':
    main()
