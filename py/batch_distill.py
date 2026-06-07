#!/usr/bin/env python3
"""Batch distillation: scales to thousands of training pairs with resume support.

Usage:
  python3 batch_distill.py --pairs 2000 --output training-data.txt
"""

import sys, os, time, json, urllib.request, argparse

API_KEY_FILE = "/home/alex/Downloads/a295bf35-eca5-4f2f-8670-d1a61ee7c7b1"
ENDPOINT = "http://localhost:8080/v1/chat/completions"
MODEL = "llama3.2-3b"
CHECKPOINT_FILE = "distill_checkpoint.json"

# ─── Seed queries ───────────────────────────────────────────────────

SEED_QUERIES = [
    "hello", "hi there", "hey", "good morning", "good evening", "howdy",
    "how are you", "what's up", "how's it going", "what's new",
    "how was your day", "nice weather today",
    "what is your name", "who are you", "what can you do",
    "tell me about yourself", "where are you from",
    "tell me a joke", "tell me a fun fact", "give me some advice",
    "what should I eat", "recommend a movie", "what's a good book",
    "how do I learn to code", "what time is it", "what day is today",
    "how do I cook pasta", "what's the capital of france",
    "what's your favorite color", "do you like music",
    "cats or dogs", "coffee or tea", "what's the meaning of life",
    "are you happy", "bye", "see you later", "goodbye", "take care",
    "thanks", "thank you", "sorry", "excuse me", "help", "I need help",
    "yes", "no", "maybe", "I don't know", "that's interesting",
    "tell me more", "I agree", "I disagree", "why", "what do you think",
    "how does that work", "can you explain that", "I don't understand",
    "that's funny", "that's sad", "wow", "cool", "awesome", "great",
    "nice", "ok", "alright", "sure", "of course", "wait", "hold on",
    "let me think", "never mind", "forget it", "I changed my mind",
    "good night", "sleep well", "have a good day", "talk soon",
]

# ─── API ────────────────────────────────────────────────────────────

def call_api(messages, max_tokens=80, temperature=0.8):
    api_key = open(API_KEY_FILE).read().strip()
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            c = data["choices"][0]["message"].get("content")
            if c is None:
                c = data["choices"][0]["message"].get("reasoning", "")
            return (c or "").strip()
        except Exception as e:
            if attempt == 2:
                return ""
            time.sleep(2)

# ─── Cleaning ───────────────────────────────────────────────────────

# Model charset: space, A-Z, 0-9, !, /, ?
ALLOWED = set(" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!/?")

def clean(text):
    """Keep only characters in the tiny model's charset."""
    return ''.join(c for c in text.upper() if c in ALLOWED).strip()

# ─── Generation ─────────────────────────────────────────────────────

def generate_query_batch():
    """Ask the teacher for 20 diverse queries."""
    prompt = (
        "Generate 20 short, natural, conversational one-line messages "
        "that a user might say to a helpful assistant. Vary topics widely. "
        "One utterance per line, no numbering, no quotes. Under 60 chars each."
    )
    try:
        text = call_api(
            [{"role": "user", "content": prompt}],
            max_tokens=500, temperature=0.95
        )
        queries = []
        for line in text.split("\n"):
            line = line.strip().strip('"').strip("'")
            line = line.lstrip("0123456789.-) ")
            line = line.strip()
            if 5 < len(line) < 80 and any(c.isalpha() for c in line):
                if not line.lower().startswith(("here", "sure", "got it", "of course")):
                    queries.append(line)
        return queries
    except Exception as e:
        print(f"  Query generation error: {e}", file=sys.stderr)
        return []

def generate_response(query):
    """Generate a natural response for a query."""
    try:
        resp = call_api(
            [{"role": "user", "content": f"Reply in 5 to 15 words, casual and friendly: {query}"}],
            max_tokens=30, temperature=0.8
        )
        resp = clean(resp)
        if len(resp) < 1 or len(resp) > 150:
            return None
        if not any(c.isalpha() for c in resp):
            return None
        return resp
    except Exception:
        return None

# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", "-n", type=int, default=2000)
    parser.add_argument("--output", "-o", default="training-data.txt")
    parser.add_argument("--checkpoint", default=CHECKPOINT_FILE)
    args = parser.parse_args()

    target = args.pairs
    output_file = args.output
    ckpt_file = args.checkpoint

    # Load or init state
    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            state = json.load(f)
        queries = state["queries"]
        pairs = state["pairs"]
        print(f"Resumed: {len(pairs)} pairs, {len(queries)} queries remaining")
    else:
        queries = list(SEED_QUERIES)
        pairs = []
        print(f"Starting fresh: {len(queries)} seed queries")

    print(f"Target: {target} pairs\n")

    # Phase 1: Expand query set via diversification
    while len(queries) < target:
        print(f"Query pool: {len(queries)}/{target} — diversifying...")
        new = generate_query_batch()
        added = [q for q in new if q.lower() not in {x.lower() for x in queries}]
        queries.extend(added)
        # Save checkpoint
        with open(ckpt_file, "w") as f:
            json.dump({"queries": queries, "pairs": pairs}, f)
        print(f"  +{len(added)} new queries (total: {len(queries)})")
        time.sleep(2)  # rate limit

    # Phase 2: Generate responses in parallel
    to_process = [q for q in queries if q.lower() not in {p[0].lower() for p in pairs}]
    print(f"\nGenerating responses for {len(to_process)} queries (4 workers)...")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    completed = 0
    
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(generate_response, q): q for q in to_process}
        
        for future in as_completed(futures):
            q = futures[future]
            try:
                r = future.result()
                if r:
                    pairs.append((q.upper(), r))
            except Exception:
                pass
            
            completed += 1
            if completed % 50 == 0 or completed == len(to_process):
                print(f"  {completed}/{len(to_process)} ({len(pairs)} valid pairs)")

    # Save checkpoint periodically during generation
    with open(ckpt_file, "w") as f:
        json.dump({"queries": queries, "pairs": pairs}, f)

    # Final write
    print(f"\n\nWriting {len(pairs)} pairs to {output_file}...")
    with open(output_file, "w") as f:
        for q, r in pairs:
            f.write(f"{q}|{r}\n")

    # Cleanup checkpoint
    os.remove(ckpt_file)
    print(f"Done! {len(pairs)} pairs written.")
    print(f"\nNext: python3 ../z80/z80ai/feedme.py --file {output_file} --epochs 300")


if __name__ == "__main__":
    main()
