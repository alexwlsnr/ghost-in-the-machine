#!/usr/bin/env python3
"""BPE tokenizer class used during training and evaluation.

Loaded from the JSON file produced by train_bpe.py.
"""
import json
from pathlib import Path
from typing import List, Tuple


class BPETokenizer:
    PAD = 0
    EOS = 1
    SEP = 2

    def __init__(self, path: str):
        data = json.loads(Path(path).read_text())
        self.vocab: dict = data['vocab']            # token_str → id
        self.id_to_token: dict = {int(k): v for k, v in data['id_to_token'].items()}
        self.merges: List[Tuple[str, str]] = [tuple(m) for m in data['merges']]
        self.merge_rank: dict = {(a, b): i for i, (a, b) in enumerate(self.merges)}
        self.vocab_size: int = data['vocab_size'] + 3   # BPE tokens + 3 special

    def encode(self, text: str) -> List[int]:
        """Encode a string into BPE token IDs (no special tokens added).

        Uses merge-rank priority: find the highest-priority (lowest-rank) adjacent
        pair, apply it, repeat until no mergeable pairs remain.
        O(n^2) on token count — far faster than O(n * M) linear scan for short texts.
        """
        pieces = list(text)
        while len(pieces) >= 2:
            best_rank = len(self.merges)
            best_i = -1
            for i in range(len(pieces) - 1):
                rank = self.merge_rank.get((pieces[i], pieces[i + 1]), len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i == -1:
                break
            pieces[best_i] = pieces[best_i] + pieces[best_i + 1]
            del pieces[best_i + 1]
        return [self.vocab.get(p, self.PAD) for p in pieces]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a string."""
        parts = []
        for i in ids:
            if i in (self.PAD, self.EOS, self.SEP):
                continue
            parts.append(self.id_to_token.get(i, ''))
        return ''.join(parts)

    def encode_pair(self, query: str, response: str) -> Tuple[List[int], List[int]]:
        """Encode a Q/R pair, returning (q_tokens, r_tokens) without special tokens."""
        return self.encode(query.upper()), self.encode(response.upper())
