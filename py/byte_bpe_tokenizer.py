#!/usr/bin/env python3
"""Drop-in replacement for BPETokenizer backed by a HuggingFace byte-level BPE.

Same interface as py/bpe_tokenizer.BPETokenizer (encode/decode/encode_pair/PAD/
EOS/SEP/vocab_size/id_to_token) so it slots into train_transformer.py unchanged,
but tokenizes with a GPT-2-style byte-level BPE that llama.cpp can reproduce.

Detected by file content: a HF tokenizer.json has a top-level "model" key.
"""
from typing import List, Tuple

from tokenizers import Tokenizer


class ByteBPETokenizer:
    def __init__(self, path: str):
        self._tk = Tokenizer.from_file(path)
        self.PAD = self._tk.token_to_id("<PAD>")
        self.EOS = self._tk.token_to_id("<EOS>")
        self.SEP = self._tk.token_to_id("<SEP>")
        assert (self.PAD, self.EOS, self.SEP) == (0, 1, 2), \
            f"expected special ids 0/1/2, got {(self.PAD, self.EOS, self.SEP)}"
        self.vocab_size = self._tk.get_vocab_size()
        vocab = self._tk.get_vocab()                 # token_str -> id
        self.id_to_token = {i: t for t, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        return self._tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i not in (self.PAD, self.EOS, self.SEP)]
        return self._tk.decode(ids)

    def encode_pair(self, query: str, response: str) -> Tuple[List[int], List[int]]:
        return self.encode(query.upper()), self.encode(response.upper())


def is_hf_tokenizer(path: str) -> bool:
    """True if the JSON looks like a HuggingFace tokenizer.json (has a 'model' key)."""
    import json
    try:
        with open(path) as f:
            d = json.load(f)
        return isinstance(d, dict) and "model" in d and "added_tokens" in d
    except Exception:
        return False
