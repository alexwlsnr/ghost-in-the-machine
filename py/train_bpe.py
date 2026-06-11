#!/usr/bin/env python3
"""Train a BPE tokenizer on the training corpus.

Uses word-frequency BPE (Sennrich et al. 2016) with space-prefixed word units
so decode is simple concatenation.  Output JSON embeds in the model manifest.

Usage:
  python3 py/train_bpe.py --data data/spec512_v12_clean.txt --vocab-size 4096
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path

# Special token IDs — BPE tokens start at 3
PAD_ID, EOS_ID, SEP_ID = 0, 1, 2
SPECIAL = {'<PAD>': PAD_ID, '<EOS>': EOS_ID, '<SEP>': SEP_ID}
SPECIAL_OFFSET = 3


def load_corpus(path: str, max_chars: int) -> str:
    text = Path(path).read_text(errors='replace')
    text = text.replace('|', ' ')   # | is handled by SEP special token
    text = text.upper()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def build_word_freqs(text: str) -> dict:
    """Split into word-units where non-first words in a line get a space prefix.
    This means decode is plain concatenation — no space re-insertion needed.
    """
    freqs: dict = {}
    for line in text.split('\n'):
        parts = line.split()
        for i, word in enumerate(parts):
            if not word:
                continue
            unit = tuple((' ' + word) if i > 0 else word)
            freqs[unit] = freqs.get(unit, 0) + 1
    return freqs


def count_pairs(freqs: dict) -> dict:
    counts: dict = {}
    for word, freq in freqs.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            counts[p] = counts.get(p, 0) + freq
    return counts


def apply_merge(freqs: dict, a: str, b: str, new_tok: str) -> dict:
    result: dict = {}
    for word, freq in freqs.items():
        if a not in word:
            result[word] = result.get(word, 0) + freq
            continue
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(new_tok)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        key = tuple(new_word)
        result[key] = result.get(key, 0) + freq
    return result


def train(text: str, vocab_size: int = 4096) -> tuple:
    # Base vocabulary: all unique chars in the corpus
    base_chars = sorted(set(c for c in text if c != '\n'))
    vocab = {c: SPECIAL_OFFSET + i for i, c in enumerate(base_chars)}
    next_id = SPECIAL_OFFSET + len(base_chars)

    print(f"  Base vocab: {len(base_chars)} chars (IDs {SPECIAL_OFFSET}–{next_id-1})")

    word_freqs = build_word_freqs(text)
    print(f"  Unique word-units: {len(word_freqs):,}")

    n_merges = vocab_size - len(vocab)
    if n_merges <= 0:
        print(f"  Warning: base vocab ({len(vocab)}) already >= vocab_size ({vocab_size})")
        return vocab, []

    print(f"  Training {n_merges} merges...")
    merges = []
    t0 = time.time()

    for step in range(n_merges):
        pairs = count_pairs(word_freqs)
        if not pairs:
            print(f"  No more pairs at step {step}")
            break
        best = max(pairs, key=pairs.__getitem__)
        best_count = pairs[best]
        if best_count < 2:
            print(f"  Min pair freq=1 at step {step}, stopping early")
            break

        a, b = best
        new_tok = a + b
        vocab[new_tok] = next_id
        merges.append([a, b])
        word_freqs = apply_merge(word_freqs, a, b, new_tok)
        next_id += 1

        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step+1}/{n_merges}: '{new_tok}' (freq={best_count}) [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"  Done: {len(merges)} merges in {elapsed:.0f}s")
    return vocab, merges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/spec512_v12_clean.txt')
    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--out', default='data/bpe_4096.json')
    parser.add_argument('--max-chars', type=int, default=10_000_000)
    args = parser.parse_args()

    print(f"Loading {args.data} (max {args.max_chars//1_000_000}M chars)...")
    text = load_corpus(args.data, args.max_chars)
    print(f"  {len(text):,} chars loaded")

    vocab, merges = train(text, args.vocab_size)

    # Build id→token for convenient TS-side decoding
    id_to_token = {v: k for k, v in vocab.items()}
    # Special tokens
    for name, tid in SPECIAL.items():
        id_to_token[tid] = name

    out = {
        'vocab_size': args.vocab_size,
        'special': {'pad': PAD_ID, 'eos': EOS_ID, 'sep': SEP_ID},
        'vocab': vocab,
        'id_to_token': {str(k): v for k, v in id_to_token.items()},
        'merges': merges,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    size_kb = Path(args.out).stat().st_size // 1024
    print(f"\nSaved {args.out} ({size_kb} KB, {len(vocab)+3} total tokens)")

    # Quick encode test
    test = "HELLO HOW ARE YOU TODAY"
    pieces = list(test)
    for a, b in merges:
        new_pieces = []
        i = 0
        while i < len(pieces):
            if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
                new_pieces.append(a + b)
                i += 2
            else:
                new_pieces.append(pieces[i])
                i += 1
        pieces = new_pieces
    ids = [vocab.get(p, 0) for p in pieces]
    decoded = ''.join(id_to_token.get(i, '?') for i in ids if i >= SPECIAL_OFFSET)
    print(f"\nEncode test: '{test}'")
    print(f"  Tokens ({len(ids)}): {pieces}")
    print(f"  Decoded: '{decoded}'")


if __name__ == '__main__':
    main()
