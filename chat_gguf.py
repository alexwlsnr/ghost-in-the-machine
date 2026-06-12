#!/usr/bin/env python3
"""Interactive chat REPL for a Ghost GGUF model via llama.cpp (stock inference).

Uses the model's native Q<SEP>R turn format and the GGUF's *embedded* byte-level
tokenizer — no external tokenizer. Multi-turn context is packed as
  Q1 <SEP> R1 <SEP> Q2 <SEP> R2 <SEP> ... Qn <SEP>
and generation stops at <SEP>/<EOS> to keep each reply clean.

Usage:
  .venv/bin/python3 chat_gguf.py [path/to/model.gguf]
Commands:  /reset  clear history   |   /quit  exit
"""
import sys

from llama_cpp import Llama

MODEL = sys.argv[1] if len(sys.argv) > 1 else "dist/gguf/wisp_bytelevel_ep21_f16.gguf"
N_CTX = 256

print(f"Loading {MODEL} ...")
llm = Llama(model_path=MODEL, n_ctx=N_CTX, verbose=False)
print("Ready. Type a message (/reset to clear, /quit to exit).\n")

history = []  # list of (q, r)


def build_prompt(turns, new_q):
    parts = []
    for q, r in turns:
        parts.append(f"{q}<SEP>{r}")
    parts.append(f"{new_q}<SEP>")
    return "<SEP>".join(parts)


while True:
    try:
        q = input("\033[1;36myou ›\033[0m ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        break
    if not q:
        continue
    if q in ("/quit", "/exit"):
        print("bye")
        break
    if q == "/reset":
        history.clear()
        print("[history cleared]\n")
        continue

    prompt = build_prompt(history, q.upper())
    # keep within context: drop oldest turns if prompt grows too long
    while len(llm.tokenize(prompt.encode(), add_bos=False, special=True)) > N_CTX - 48 and history:
        history.pop(0)
        prompt = build_prompt(history, q.upper())

    print("\033[1;32mghost ›\033[0m ", end="", flush=True)
    reply = ""
    for chunk in llm.create_completion(prompt, max_tokens=120, temperature=0.6,
                                       top_k=20, repeat_penalty=1.3,
                                       stop=["<SEP>", "<EOS>"], stream=True):
        piece = chunk["choices"][0]["text"]
        reply += piece
        print(piece, end="", flush=True)
    print("\n")
    history.append((q.upper(), reply.strip()))
