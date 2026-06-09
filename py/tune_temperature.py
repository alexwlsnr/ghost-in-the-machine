#!/usr/bin/env python3
"""
Temperature sweep to find optimal per-model temperature.

Generates responses at each temperature and scores for coherence.
Run: .venv/bin/python3 py/tune_temperature.py
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from train_transformer import (
    TinyTransformer, TinyTransformerModern, TinyTransformerTernary,
    encode, SEP_TOKEN, PAD_TOKEN, EOS_TOKEN,
)

PROMPTS = [
    "HELLO", "WHO ARE YOU", "HOW ARE YOU", "TELL ME A JOKE",
    "I FEEL SAD TODAY", "WHAT CAN YOU DO", "GOOD MORNING", "THANKS",
]

TEMPS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]

def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    arch = ck['architecture']
    if arch.get('arch') == 'ternary':
        model = TinyTransformerTernary(**{k: arch[k] for k in ['vocab_size','d_model','n_heads','n_layers','d_ff','max_len']})
    elif arch.get('arch') == 'modern':
        state = ck['model_state']
        model = TinyTransformerModern(
            vocab_size=arch['vocab_size'], d_model=arch['d_model'],
            n_heads=arch['n_heads'], n_layers=arch['n_layers'],
            d_ff=arch['d_ff'], max_len=arch['max_len'],
            use_rope='pos_embed.weight' not in state,
            use_swiglu=any('ff.w1' in k for k in state),
            use_rmsnorm=not any('norm1.bias' in k for k in state),
            tie_weights='head.weight' not in state,
        )
    else:
        model = TinyTransformer(**{k: arch[k] for k in ['vocab_size','d_model','n_heads','n_layers','d_ff','max_len']})
    model.load_state_dict(ck['model_state'])
    model.eval()
    return model, arch

def generate(model, prompt, temp, max_new=50, seed=42):
    torch.manual_seed(seed)
    toks = [*encode(prompt.upper()), SEP_TOKEN]
    with torch.no_grad():
        for _ in range(max_new):
            if len(toks) >= model.max_len - 1: break
            x = torch.tensor([toks], dtype=torch.long)
            logits = model(x)[0, -1] / max(temp, 0.01)
            probs = F.softmax(logits, dim=-1)
            next_tok = int(torch.multinomial(probs, 1))
            if next_tok in (EOS_TOKEN, PAD_TOKEN): break
            toks.append(next_tok)
    return ''.join(chr(t) for t in toks[len(encode(prompt.upper()))+1:] if 32 <= t < 127)

def score(resp, prompt):
    """Simple coherence score 0-10."""
    if not resp or len(resp) < 4: return 0
    # Penalise repetition: same char 4+ times in a row
    if re.search(r'(.)\1{3,}', resp): return 1
    # Penalise garbled strings (too many consonant clusters)
    words = resp.split()
    garbled = sum(1 for w in words if len(w) > 3 and not re.search(r'[AEIOU]', w))
    if garbled > len(words) * 0.4: return 2
    # Base score from length + English-ness
    score = min(len(resp) / 8, 5)
    # Bonus for natural punctuation
    if any(c in resp for c in '!?.,\''): score += 1
    # Bonus for persona keywords
    if any(w in resp for w in ['GHOST','HUMAN','AI','BYTE']): score += 1
    # Penalty for very short non-answers
    if len(resp) < 8 and resp.strip() in ('HEY THERE!','HI!','HELLO!'): score *= 0.7
    return min(score, 10)

def sweep(name, path):
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    model, arch = load_model(path)
    vl = torch.load(path, map_location='cpu', weights_only=False).get('best_val_loss', '?')
    print(f"Val loss: {vl:.4f}  d={arch['d_model']} L={arch['n_layers']}")

    best_temp, best_score = 0.7, 0
    results = []
    for temp in TEMPS:
        scores = []
        for prompt in PROMPTS:
            resp = generate(model, prompt, temp)
            scores.append(score(resp, prompt))
        avg = sum(scores) / len(scores)
        results.append((temp, avg))
        if avg > best_score:
            best_score = avg
            best_temp = temp

    print(f"\n{'Temp':>6}  {'Score':>6}")
    for t, s in results:
        marker = ' ◄ best' if t == best_temp else ''
        print(f"{t:>6.1f}  {s:>6.2f}{marker}")

    # Sample outputs at best temp
    print(f"\nSample outputs at temp={best_temp}:")
    for prompt in PROMPTS[:4]:
        resp = generate(model, prompt, best_temp)
        print(f"  {prompt:20s} → {resp[:55]}")

    return best_temp

MODELS = [
    ("Wisp fp32",    "ckpt/wisp_shade_data.pt"),
    ("Shade compact","ckpt/shade_compact.pt"),
    ("Spec512 v1.1", "ckpt/spec512_v2.pt"),
    ("Wisp ternary", "ckpt/wisp_ternary_final.pt"),
]

print("Temperature sweep across models")
recommendations = {}
for name, path in MODELS:
    if not os.path.exists(path):
        print(f"\nSkipping {name} — {path} not found")
        continue
    best = sweep(name, path)
    recommendations[name] = best

print(f"\n{'='*60}")
print("RECOMMENDATIONS:")
for name, temp in recommendations.items():
    print(f"  {name:20s}: temp = {temp}")
