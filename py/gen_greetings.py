#!/usr/bin/env python3
"""
Generate a high-coverage greeting dataset for Wisp and Shade.

Greetings are the most common first prompt and the smallest models
handle variations poorly. This script generates hundreds of greeting
variations programmatically (no teacher needed for prompts) then
calls the teacher for natural responses.

Usage:
  python3 py/gen_greetings.py --output data/greetings.txt --workers 32
"""

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill import chat_completion

# ── Greeting building blocks ──────────────────────────────────────────────────

BASE = [
    "hello", "hi", "hey", "howdy", "greetings", "yo",
    "sup", "hiya", "heya", "what's up", "whats up",
    "good morning", "good afternoon", "good evening", "good day",
    "morning", "evening",
    "hi there", "hello there", "hey there",
    "hey you", "hello you",
]

# Suffixes that can follow a greeting
SUFFIXES = [
    "", "!", "?", "!!", "...", " :)",
    " friend", " there", " mate", " pal",
    " how are you", " how are you doing", " how's it going",
    " what's up", " whats up", " what's good",
    " I'm new here", " just wanted to say hi",
    " nice to meet you", " good to see you",
    " can you hear me", " is anyone there",
    " i need help", " i have a question",
    " claude", " ai", " bot", " robot",
    " nice bot", " cool ai",
]

# Standalone greeting phrases (not derived from BASE)
STANDALONE = [
    "yo what's good",
    "ayo",
    "salutations",
    "ahoy",
    "wassup",
    "wazzup",
    "sup man",
    "hey friend",
    "hello hello",
    "hi hi",
    "hey hey",
    "oh hello",
    "oh hi",
    "oh hey",
    "well hello there",
    "well hi there",
    "hello world",
    "hi world",
    "greetings and salutations",
    "good to meet you",
    "pleased to meet you",
    "nice to meet you",
    "how do you do",
    "how are you",
    "how are you doing",
    "how are you today",
    "how's it going",
    "how's everything",
    "how's life",
    "what's new",
    "what's happening",
    "what's going on",
    "long time no see",
    "it's been a while",
    "I'm back",
    "I'm here",
    "testing testing",
    "hello are you there",
    "is this thing on",
    "anyone home",
    "knock knock",
    "good to see you again",
]


def build_prompts():
    seen = set()
    prompts = []

    def add(p):
        p = p.strip()
        if not p or len(p) > 60:  # keep prompts short — must fit ctx=64 with response
            return
        norm = p.upper()
        if norm not in seen:
            seen.add(norm)
            prompts.append(norm)

    # Base × suffix combinations
    for base, suffix in itertools.product(BASE, SUFFIXES):
        add(base + suffix)

    # Standalone phrases
    for s in STANDALONE:
        add(s)

    return prompts


SYSTEM_PROMPT = (
    "You are a friendly, warm AI assistant having a casual conversation. "
    "When someone greets you, respond with a natural, brief greeting back. "
    "Keep responses under 60 characters. "
    "Never ask follow-up questions. Never start with 'I'. "
    "Just say hello back in a varied, natural way."
)


def generate_response(endpoint, model, query, api_key=""):
    try:
        resp = chat_completion(
            endpoint, model,
            messages=[{"role": "user", "content": f"Respond to: {query}"}],
            max_tokens=40,
            temperature=0.9,
            api_key=api_key,
            system=SYSTEM_PROMPT,
        )
        resp = resp.strip().strip('"').strip("'")
        resp = ''.join(c for c in resp if c.isascii() and ord(c) >= 32).strip()
        if len(resp) < 2 or len(resp) > 80:
            return None
        if not any(c.isalpha() for c in resp):
            return None
        return resp.upper()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate greeting training pairs")
    parser.add_argument("--output", "-o", default="data/greetings.txt")
    parser.add_argument("--endpoint", "-e", default="http://localhost:8080/v1")
    parser.add_argument("--model", "-m", default="gemma4-e4b-distill")
    parser.add_argument("--workers", "-w", type=int, default=32)
    parser.add_argument("--responses-per-prompt", type=int, default=3,
                        help="Generate N response variants per prompt (diversity)")
    parser.add_argument("--api-key", "-k", default=None)
    args = parser.parse_args()

    api_key = ""
    if args.api_key:
        api_key = open(args.api_key).read().strip() if os.path.isfile(args.api_key) else args.api_key

    prompts = build_prompts()
    print(f"Generated {len(prompts)} unique greeting prompts")

    # Expand: N responses per prompt for diversity
    tasks = [(q, i) for q in prompts for i in range(args.responses_per_prompt)]
    print(f"Generating {len(tasks)} responses ({args.responses_per_prompt} per prompt) "
          f"via {args.workers} workers...")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    pairs = []
    errors = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_response, args.endpoint, args.model, q, api_key): q
            for q, _ in tasks
        }
        for future in as_completed(futures):
            q = futures[future]
            completed += 1
            resp = future.result()
            if resp and len(q) + len(resp) + 1 <= 128:
                pairs.append((q, resp))
            else:
                errors += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"  {completed}/{len(tasks)} done  {len(pairs)} valid pairs")

    # Deduplicate by (query, response) and write
    seen_pairs = set()
    deduped = []
    for q, r in pairs:
        key = (q, r[:20])  # dedupe by query + response prefix
        if key not in seen_pairs:
            seen_pairs.add(key)
            deduped.append((q, r))

    wisp  = [(q, r) for q, r in deduped if len(q) + len(r) + 1 <= 64]
    shade = deduped  # all fit ctx=128

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        for q, r in deduped:
            f.write(f"{q}|{r}\n")

    print(f"\nWritten {len(deduped)} pairs → {args.output}")
    print(f"  Wisp-compatible  (ctx≤64):  {len(wisp)}")
    print(f"  Shade-compatible (ctx≤128): {len(shade)}")
    print(f"\nMix into training data:")
    print(f"  cat data/greetings.txt py/training-data-transformer.txt > data/wisp_train_v2.txt")


if __name__ == "__main__":
    main()
