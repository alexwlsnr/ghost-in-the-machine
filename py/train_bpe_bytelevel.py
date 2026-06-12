#!/usr/bin/env python3
"""Train a GPT-2-style byte-level BPE tokenizer (llama.cpp-compatible).

Unlike py/train_bpe.py (char-level, no pretokenizer, global-greedy merges), this
produces a standard HuggingFace tokenizer.json with:
  - ByteLevel pre-tokenizer + GPT-2 regex split (use_regex=True)
  - byte-level alphabet (all 256 bytes mapped to printable unicode)
  - merges in byte-mapped form

This is exactly the family llama.cpp's `gpt2` tokenizer reproduces, so a model
trained on this tokenizer round-trips faithfully into stock llama-cli/ollama.

Special tokens <PAD>=0, <EOS>=1, <SEP>=2 to match the existing arch convention.
"""
import argparse

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab-size", type=int, default=4099,
                    help="total vocab incl. 3 specials + 256 bytes (match arch)")
    ap.add_argument("--out", required=True, help="output tokenizer.json")
    args = ap.parse_args()

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<PAD>", "<EOS>", "<SEP>"],   # -> ids 0,1,2
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        show_progress=True,
    )
    tk.train([args.data], trainer)
    tk.save(args.out)

    vs = tk.get_vocab_size()
    print(f"[bytelevel-bpe] trained vocab={vs}  saved {args.out}")
    for s in ["<PAD>", "<EOS>", "<SEP>"]:
        print(f"  {s} -> id {tk.token_to_id(s)}")
    # sanity round-trip
    sample = "HELLO THERE FRIEND, HOW ARE YOU?"
    enc = tk.encode(sample)
    print(f"  sample ids   : {enc.ids[:16]}")
    print(f"  sample pieces: {enc.tokens[:16]}")
    print(f"  round-trip   : {tk.decode(enc.ids)!r}")


if __name__ == "__main__":
    main()
