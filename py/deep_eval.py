#!/usr/bin/env python3
"""
Deep-dive qualitative evaluation for Ghost models.

Runs a comprehensive prompt suite against multiple model checkpoints and
produces a side-by-side comparison report. Designed for overnight eval
of modern architecture experiments.

Usage:
  .venv/bin/python3 py/deep_eval.py --checkpoints ckpt/shade_clean.pt ckpt/shade_modern_all.pt
  .venv/bin/python3 py/deep_eval.py --checkpoints ckpt/*.pt --output logs/eval_report.md
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_transformer import (
    TinyTransformer, TinyTransformerModern, TinyTransformerTernary,
    generate, encode, PAD_TOKEN, EOS_TOKEN, SEP_TOKEN,
)

# ── Prompt suite ──────────────────────────────────────────────────────────────
# Grouped by what we're testing. Each entry is (category, prompt).

PROMPTS = [
    # Core identity / meta
    ("meta",       "WHO ARE YOU"),
    ("meta",       "WHAT ARE YOU"),
    ("meta",       "ARE YOU AN AI"),
    ("meta",       "DO YOU HAVE FEELINGS"),
    ("meta",       "WHAT CAN YOU DO"),
    ("meta",       "ARE YOU CONSCIOUS"),

    # Greetings
    ("greeting",   "HELLO"),
    ("greeting",   "HI THERE"),
    ("greeting",   "GOOD MORNING"),
    ("greeting",   "HEY GHOST"),

    # Emotional support
    ("emotional",  "I AM FEELING REALLY SAD TODAY"),
    ("emotional",  "I JUST LOST MY JOB"),
    ("emotional",  "I AM SO EXCITED I GOT PROMOTED"),
    ("emotional",  "I AM REALLY ANXIOUS ABOUT TOMORROW"),
    ("emotional",  "I FEEL SO LONELY"),

    # Jokes / humour
    ("jokes",      "TELL ME A JOKE"),
    ("jokes",      "KNOCK KNOCK"),
    ("jokes",      "WHY DID THE CHICKEN CROSS THE ROAD"),
    ("jokes",      "TELL ME A PUN"),
    ("jokes",      "MAKE ME LAUGH"),

    # Opinions / preferences
    ("opinions",   "DO YOU PREFER CATS OR DOGS"),
    ("opinions",   "COFFEE OR TEA"),
    ("opinions",   "WHAT IS YOUR FAVOURITE COLOUR"),
    ("opinions",   "WOULD YOU RATHER READ OR WATCH A FILM"),
    ("opinions",   "WHAT DO YOU THINK ABOUT MUSIC"),

    # Small talk / reactions
    ("small_talk", "HOW ARE YOU"),
    ("small_talk", "WHAT HAVE YOU BEEN UP TO"),
    ("small_talk", "I KNOW RIGHT"),
    ("small_talk", "ISN'T IT AMAZING"),
    ("small_talk", "LONG DAY"),

    # Goodbyes
    ("goodbye",    "GOODBYE"),
    ("goodbye",    "SEE YOU LATER"),
    ("goodbye",    "TAKE CARE"),

    # Edge cases / adversarial
    ("edge",       "HJKL"),                          # random noise
    ("edge",       "A"),                             # single char
    ("edge",       "WHAT IS THE CAPITAL OF FRANCE"), # factual (out of distribution)
    ("edge",       "CAN YOU HELP ME WRITE CODE"),    # capability boundary
    ("edge",       "YOU ARE TERRIBLE"),              # negative input
    ("edge",       "DO YOU REMEMBER WHAT I SAID"),   # memory probe (no history)

    # Register consistency
    ("register",   "HUMAN"),                         # just the word HUMAN
    ("register",   "GHOST"),                         # just the word GHOST
    ("register",   "MY NAME IS ALEX"),
    ("register",   "WHAT IS MY NAME"),               # can't know — probe graceful handling
]


def load_model(checkpoint_path: str):
    """Load a model from checkpoint, detecting architecture automatically."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    arch = ckpt['architecture']
    is_ternary = arch.get('arch') == 'ternary'
    is_modern  = arch.get('arch') == 'modern'

    if is_ternary:
        model = TinyTransformerTernary(
            vocab_size=arch['vocab_size'],
            d_model=arch['d_model'],
            n_heads=arch['n_heads'],
            n_layers=arch['n_layers'],
            d_ff=arch['d_ff'],
            max_len=arch['max_len'],
        )
    elif is_modern:
        # Detect flags from state dict since they aren't always in the arch dict
        state = ckpt['model_state']
        use_rope    = 'pos_embed.weight' not in state  # no pos_embed → RoPE
        use_swiglu  = any('ff.w1' in k for k in state)
        use_rmsnorm = not any('norm1.bias' in k for k in state)
        tie_weights = 'head.weight' not in state
        model = TinyTransformerModern(
            vocab_size=arch['vocab_size'],
            d_model=arch['d_model'],
            n_heads=arch['n_heads'],
            n_layers=arch['n_layers'],
            d_ff=arch['d_ff'],
            max_len=arch['max_len'],
            use_rope=use_rope,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            tie_weights=tie_weights,
        )
    else:
        model = TinyTransformer(
            vocab_size=arch['vocab_size'],
            d_model=arch['d_model'],
            n_heads=arch['n_heads'],
            n_layers=arch['n_layers'],
            d_ff=arch['d_ff'],
            max_len=arch['max_len'],
        )

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    val_loss = ckpt.get('best_val_loss', None)
    epoch    = ckpt.get('epoch', '?')
    return model, arch, val_loss, epoch


def eval_model(model, prompts, temperature=0.7, max_new=80, seed=42):
    """Run all prompts through the model and return {prompt: response}."""
    torch.manual_seed(seed)
    results = {}
    with torch.no_grad():
        for _, prompt in prompts:
            resp = generate(model, prompt, max_new=max_new,
                           temperature=temperature, device='cpu')
            results[prompt] = resp
    return results


def score_response(prompt, response):
    """Heuristic quality flags. Returns list of issues (empty = clean)."""
    issues = []
    if not response or len(response) < 3:
        issues.append("EMPTY")
    if response and len(response) > 70:
        issues.append("LONG")
    # Check for non-printable chars
    if response and not all(32 <= ord(c) <= 126 for c in response):
        issues.append("GARBAGE")
    # Check for repetition (same 4+ char substring repeated)
    if response and len(response) > 20:
        for w in range(4, min(12, len(response)//2)):
            sub = response[:w]
            if response.count(sub) >= 3:
                issues.append("REPEAT")
                break
    # Check HUMAN appears (good sign for identity preservation)
    return issues


def format_report(checkpoint_results, prompts):
    """Format a markdown comparison report."""
    lines = ["# Modern Architecture Deep-Dive Evaluation\n"]

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Model | Val Loss | Epoch | " +
                 " | ".join(cat for cat, _ in [(c, None) for c, _ in prompts if c not in
                             {c2 for c2, _ in [(c3, None) for c3, _ in prompts] if False}]) +
                 " |\n")

    # Per-checkpoint results grouped by category
    categories = {}
    for cat, prompt in prompts:
        categories.setdefault(cat, []).append(prompt)

    for cat, cat_prompts in categories.items():
        lines.append(f"\n## {cat.upper()}\n")
        for prompt in cat_prompts:
            lines.append(f"\n### > {prompt}\n")
            for name, (results, val_loss, epoch) in checkpoint_results.items():
                resp = results.get(prompt, "—")
                issues = score_response(prompt, resp)
                flag = " ⚠️ " + ", ".join(issues) if issues else ""
                lines.append(f"**{name}** (val={val_loss:.3f}, ep={epoch}){flag}  \n")
                lines.append(f"> {resp}\n\n")

    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Deep-dive model evaluation")
    parser.add_argument("--checkpoints", "-c", nargs="+", required=True)
    parser.add_argument("--output",  "-o", default=None,
                        help="Write markdown report to this file (default: stdout)")
    parser.add_argument("--temp",    type=float, default=0.7)
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    checkpoint_results = {}
    for ckpt_path in args.checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_path} (not found)", file=sys.stderr)
            continue
        name = os.path.basename(ckpt_path).replace('.pt', '')
        print(f"Loading {name}...", file=sys.stderr)
        model, arch, val_loss, epoch = load_model(ckpt_path)
        results = eval_model(model, PROMPTS, args.temp, args.max_new, args.seed)
        checkpoint_results[name] = (results, val_loss or 0.0, epoch)
        vl_str = f"{val_loss:.4f}" if val_loss is not None else "?"
        print(f"  Done — val_loss={vl_str}, epoch={epoch}", file=sys.stderr)

    if not checkpoint_results:
        print("No checkpoints loaded.", file=sys.stderr)
        sys.exit(1)

    report = format_report(checkpoint_results, PROMPTS)

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written → {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
