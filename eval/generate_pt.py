#!/usr/bin/env python3
"""Fast PyTorch eval-output generator (source-of-truth, no WASM stack).

    python eval/generate_pt.py <checkpoint.pt> <tokenizer.json> <out_tag> [--temp 0.8] [--seed 1234] [--device cpu] [--set eval_set.jsonl]

Loads a .pt checkpoint (reconstructing the arch exactly as the trainer's resume
path does — ternary forward applies BitNet quantization, so this is faithful to
the served model), then runs the trainer's generate() over the eval set and
writes eval/out_<tag>.jsonl. Far faster than the node engine (GPU/CPU PyTorch,
no O(n^2) JS token loop).
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
import train_transformer as T
import byte_bpe_tokenizer as _byte

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    kw = dict(vocab_size=arch["vocab_size"], d_model=arch["d_model"],
              n_heads=arch["n_heads"], n_layers=arch["n_layers"],
              d_ff=arch["d_ff"], max_len=arch["max_len"])
    a = arch.get("arch")
    if a == "ternary_modern":
        model = T.TinyTransformerTernaryModern(**kw, ffn_type=arch.get("ffn_type", "swiglu"),
                                               n_kv_heads=arch.get("n_kv_heads"))
    elif a == "ternary":
        model = T.TinyTransformerTernary(**kw)
    elif a == "modern":
        state = ckpt["model_state"]
        model = T.TinyTransformerModern(**kw, use_rope="pos_embed.weight" not in state,
                                        use_swiglu=any("ff.w1" in k for k in state),
                                        use_rmsnorm=not any("norm1.bias" in k for k in state),
                                        tie_weights="head.weight" not in state)
    else:
        model = T.TinyTransformer(**kw)
    model.load_state_dict(ckpt["model_state"])
    return model, arch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("tokenizer"); ap.add_argument("out_tag")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", default="eval_set.jsonl")
    ap.add_argument("--max-new", type=int, default=80)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    tok = _byte.ByteBPETokenizer(a.tokenizer) if _byte.is_hf_tokenizer(a.tokenizer) \
        else T.BPETokenizer(a.tokenizer)
    model, arch = load_model(a.checkpoint, a.device)
    print(f"loaded {a.checkpoint} ({arch.get('arch')}, n_kv_heads={arch.get('n_kv_heads')}) "
          f"on {a.device}, vocab={tok.vocab_size}", file=sys.stderr)

    set_path = os.path.join(os.path.dirname(__file__), a.set)
    rows = [json.loads(l) for l in open(set_path) if l.strip()]
    out = []
    for r in rows:
        resp = T.generate(model, r["prompt"], max_new=a.max_new, temperature=a.temp,
                          device=a.device, tok=tok)
        out.append({"id": r["id"], "category": r["category"], "prompt": r["prompt"], "response": resp})
        print(f"  [{r['id']}] {resp[:60]}", file=sys.stderr)

    dest = os.path.join(os.path.dirname(__file__), f"out_{a.out_tag}.jsonl")
    with open(dest, "w") as f:
        f.write("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"wrote {dest} ({len(out)} responses)", file=sys.stderr)

if __name__ == "__main__":
    main()
