#!/usr/bin/env python3
"""End-to-end generation: llama.cpp runs the GGUF, our BPE tokenizer does encode/decode.

Proves the converted model produces coherent text under llama.cpp inference. The
tokenizer is applied in Python (our exact char-level BPE) and token IDs are fed to
llama.cpp directly, sidestepping llama.cpp's incompatible gpt2 pretokenizer.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "py")
from bpe_tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="HELLO, HOW ARE YOU?")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.6)
    args = ap.parse_args()

    tok = BPETokenizer(args.tokenizer)
    from llama_cpp import Llama
    llm = Llama(model_path=args.gguf, n_ctx=256, logits_all=True, verbose=False)

    ids = tok.encode(args.prompt.upper()) + [tok.SEP]
    out_ids = []
    rng = np.random.default_rng(0)
    for _ in range(args.max_new):
        seq = ids + out_ids
        llm.reset()
        llm.eval(seq)
        logits = np.array(llm.scores[len(seq) - 1])
        if args.temp <= 0:
            nxt = int(logits.argmax())
        else:
            p = np.exp((logits - logits.max()) / args.temp)
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
        if nxt == tok.EOS:
            break
        out_ids.append(nxt)

    print(f"PROMPT : {args.prompt}")
    print(f"REPLY  : {tok.decode(out_ids)}")


if __name__ == "__main__":
    main()
