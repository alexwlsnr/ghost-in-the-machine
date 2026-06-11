#!/usr/bin/env python3
"""
Side-by-side subjective comparison of two models.

Usage:
  .venv/bin/python3 py/compare_models.py
"""
import sys, torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bpe_tokenizer import BPETokenizer

PROMPTS = [
    "HELLO",
    "HEY",
    "HOW ARE YOU?",
    "TELL ME A JOKE",
    "WHAT IS A BUS?",
    "WHY DOES IT RAIN?",
    "TELL ME ABOUT YOURSELF",
    "I'M FEELING SAD TODAY",
    "I GOT A PROMOTION!",
    "WHAT SHOULD I DO THIS WEEKEND?",
    "CAN YOU HELP ME?",
    "THANKS",
]

MULTI_TURN = [
    ("HOW ARE YOU?", "I AM GOOD THANKS. WHAT IS YOUR NAME?"),
    ("TELL ME A JOKE", "THAT WAS FUNNY! TELL ME ANOTHER ONE"),
]


# ── Byte-level model (classic) ─────────────────────────────────────────────────

def load_byte_model(ckpt_path, device='cpu'):
    from train_transformer import TinyTransformer, TinyTransformerModern
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck.get('architecture', ck.get('config', {}))
    arch_type = a.get('arch', a.get('type', 'classic'))
    cls = TinyTransformerModern if arch_type == 'modern' else TinyTransformer
    model = cls(
        vocab_size=a['vocab_size'], d_model=a['d_model'],
        n_heads=a['n_heads'], n_layers=a['n_layers'],
        d_ff=a['d_ff'], max_len=a['max_len'],
    )
    state = {k.replace('_orig_mod.', ''): v for k, v in ck['model_state'].items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device), a


def generate_byte(model, prompt, *, max_new=120, temperature=0.8, device='cpu') -> str:
    from train_transformer import generate
    return generate(model, prompt, max_new=max_new, temperature=temperature, device=device)


# ── BPE ternary model ──────────────────────────────────────────────────────────

def load_bpe_model(ckpt_path, device='cpu'):
    from train_transformer import TinyTransformerTernary
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck['architecture']
    model = TinyTransformerTernary(
        vocab_size=a['vocab_size'], d_model=a['d_model'],
        n_heads=a['n_heads'], n_layers=a['n_layers'],
        d_ff=a['d_ff'], max_len=a['max_len'],
    )
    state = {k.replace('_orig_mod.', ''): v for k, v in ck['model_state'].items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device), a


def generate_bpe(model, tok, prompt, *, max_new=100, temperature=0.6,
                 top_k=0, top_p=0.0, rep_penalty=1.35,
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
                    next_logits[tid] = next_logits[tid] / rep_penalty if next_logits[tid] > 0 else next_logits[tid] * rep_penalty
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < topk_vals[-1]] = -float('inf')
            if 0.0 < top_p < 1.0:
                sl, si = torch.sort(next_logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cp > top_p] = -float('inf')
                next_logits = torch.zeros_like(next_logits).scatter_(0, si, sl)
            if torch.all(next_logits == -float('inf')):
                next_token = torch.argmax(logits[0, -1]).item()
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(torch.clamp(probs, min=0), 1).item()
            if next_token in (tok.EOS, tok.PAD, tok.SEP):
                break
    return tok.decode(generated)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading models...", flush=True)
    byte_model, byte_cfg = load_byte_model('ckpt/shade_compact.pt')
    bpe_model,  bpe_cfg  = load_bpe_model('ckpt/shade_bpe_ternary.pt')
    tok = BPETokenizer('data/bpe_4096.json')

    print(f"Shade BF16   : d={byte_cfg['d_model']} {byte_cfg['n_layers']}L vocab={byte_cfg['vocab_size']} val={byte_cfg.get('best_val_loss',bpe_cfg.get('best_val_loss','?'))}")
    print(f"Shade BPE T  : d={bpe_cfg['d_model']} {bpe_cfg['n_layers']}L vocab={bpe_cfg['vocab_size']}")
    print()

    W = 52  # column width

    def row(label, a, b):
        print(f"  {'BF16':6s}  {a[:W]}")
        print(f"  {'BPE-T':6s}  {b[:W]}")
        print()

    print("═" * 80)
    print(f"{'PROMPT':<28}  {'SHADE BF16':<38}  SHADE BPE TERNARY")
    print("═" * 80)

    for prompt in PROMPTS:
        a = generate_byte(byte_model, prompt, temperature=0.8)
        b = generate_bpe(bpe_model, tok, prompt, temperature=0.6, rep_penalty=1.35)
        print(f"\n>>> {prompt}")
        print(f"  BF16   {a[:W+10]}")
        print(f"  BPE-T  {b[:W+10]}")

    print("\n" + "═" * 80)
    print("MULTI-TURN")
    print("═" * 80)
    for turns in MULTI_TURN:
        print(f"\n>>> Turn 1: {turns[0]}")
        a1 = generate_byte(byte_model, turns[0], temperature=0.8)
        b1 = generate_bpe(bpe_model, tok, turns[0], temperature=0.6, rep_penalty=1.35)
        print(f"  BF16   {a1[:W+10]}")
        print(f"  BPE-T  {b1[:W+10]}")

        sep = '|'
        ctx_a = turns[0] + sep + a1 + sep + turns[1]
        ctx_b = turns[0] + sep + b1 + sep + turns[1]
        print(f">>> Turn 2: {turns[1]}")
        a2 = generate_byte(byte_model, ctx_a, temperature=0.8)
        b2 = generate_bpe(bpe_model, tok, ctx_b, temperature=0.6, rep_penalty=1.35)
        print(f"  BF16   {a2[:W+10]}")
        print(f"  BPE-T  {b2[:W+10]}")


if __name__ == '__main__':
    main()
