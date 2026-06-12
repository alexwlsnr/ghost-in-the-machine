#!/usr/bin/env python3
"""Verify a GGUF conversion by comparing logits against the source PyTorch model.

Feeds the SAME token-id sequence to both (bypassing tokenization entirely), so this
isolates the weight + compute-graph conversion from any tokenizer mismatch.
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "py")
from train_transformer import TinyTransformerTernaryModern


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gguf", required=True)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["architecture"]
    model = TinyTransformerTernaryModern(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"], max_len=cfg["max_len"],
    )
    model.load_state_dict(ck["model_state"])
    model.eval()

    # arbitrary but valid token-id sequence (avoid special ids 0/1/2)
    tokens = [10, 55, 200, 17, 333, 4, 90, 1001, 42, 7, 512, 88]
    tokens = [t % cfg["vocab_size"] for t in tokens]

    with torch.no_grad():
        pt_logits = model(torch.tensor([tokens])).squeeze(0).float().numpy()  # (T, vocab)

    from llama_cpp import Llama
    llm = Llama(model_path=args.gguf, n_ctx=256, logits_all=True, verbose=False)
    llm.reset()
    llm.eval(tokens)
    gg_logits = np.array(llm.scores[: len(tokens)])  # (T, vocab)

    print(f"shapes: pytorch {pt_logits.shape}  gguf {gg_logits.shape}")
    pt_arg = pt_logits.argmax(-1)
    gg_arg = gg_logits.argmax(-1)
    agree = (pt_arg == gg_arg).mean()
    # per-position correlation of the full logit vectors
    cors = [np.corrcoef(pt_logits[t], gg_logits[t])[0, 1] for t in range(len(tokens))]
    maxdiff = np.abs(pt_logits - gg_logits).max()

    print(f"argmax agreement : {agree*100:.1f}%  ({(pt_arg==gg_arg).sum()}/{len(tokens)})")
    print(f"mean logit corr  : {np.mean(cors):.5f}")
    print(f"max abs logit diff: {maxdiff:.4f}")
    print(f"\npytorch next-token argmax: {pt_arg.tolist()}")
    print(f"gguf    next-token argmax: {gg_arg.tolist()}")

    if agree == 1.0 and np.mean(cors) > 0.99:
        print("\n✅ PASS — conversion is numerically faithful")
    elif agree >= 0.8 and np.mean(cors) > 0.95:
        print("\n⚠️  CLOSE — mostly matches; likely f16 rounding or minor convention diff")
    else:
        print("\n❌ FAIL — logits diverge; check rope type / q-k permute / scale fold")


if __name__ == "__main__":
    main()
