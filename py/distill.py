#!/usr/bin/env python3
"""
Self-distillation pipeline — generate training data for the tiny Z80-μLM
by querying a larger teacher model.

Produces query|response pairs in the format feedme.py expects.

Usage:
  python3 distill.py --endpoint http://localhost:8080/v1 --model gpt-oss-120b --pairs 5000
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─── Prompt templates for diverse, conversational training data ────

SEED_QUERIES = [
    # Greetings
    "hello", "hi there", "hey", "good morning", "good evening", "howdy",
    # Small talk
    "how are you", "what's up", "how's it going", "what's new",
    "how was your day", "nice weather today",
    # Questions
    "what is your name", "who are you", "what can you do",
    "tell me about yourself", "where are you from",
    # Requests
    "tell me a joke", "tell me a fun fact", "give me some advice",
    "what should I eat for dinner", "recommend a movie",
    "what's a good book to read", "how do I learn to code",
    # Practical
    "what time is it", "what day is today",
    "what's the weather", "how do I cook pasta",
    "how do I fix a flat tire", "what's the capital of France",
    # Opinions
    "what's your favorite color", "do you like music",
    "cats or dogs", "coffee or tea",
    "what's the meaning of life", "are you happy",
    # Goodbyes
    "bye", "see you later", "goodbye", "take care",
    "have a good night", "talk to you soon",
    # Varied
    "thanks", "thank you", "sorry", "excuse me",
    "help", "I need help", "can you help me",
    "yes", "no", "maybe", "I don't know",
    "that's interesting", "tell me more",
    "I agree", "I disagree", "why",
    "what do you think", "how does that work",
    "can you explain that", "I don't understand",
    "that's funny", "that's sad", "wow",
    "cool", "awesome", "great", "nice",
    "ok", "alright", "sure", "of course",
    "wait", "hold on", "let me think",
    "never mind", "forget it", "I changed my mind",
]

# Diversity prompts to generate MORE queries from the teacher
DIVERSIFY_PROMPT = """Generate 20 short, natural, conversational one-line queries or statements that a user might say to a helpful assistant.

Rules:
- Each line should be a complete, standalone utterance
- Vary the topics widely (greetings, questions, requests, opinions, small talk, goodbyes)
- Keep each line under 60 characters
- Use casual, natural language
- Do NOT number them or add any prefix — just one utterance per line
- Include some with typos or informal language for realism

Output exactly 20 lines:"""

# Template for generating a single response
RESPONSE_PROMPT = """You are a friendly, helpful assistant having a casual conversation. 
Respond to this message naturally and concisely (10-50 characters is ideal, but use what feels natural):

User: {query}
Assistant:"""


def chat_completion(endpoint: str, model: str, messages: list,
                    max_tokens: int = 80, temperature: float = 0.8,
                    api_key: str = "") -> str:
    """Call an OpenAI-compatible chat completions API."""
    url = endpoint.rstrip("/")
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"].get("content", "")
            if content is None:
                content = data["choices"][0]["message"].get("reasoning", "")
            return (content or "").strip()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def generate_response(endpoint: str, model: str, query: str, api_key: str = "") -> str | None:
    """Generate a single response for a query."""
    try:
        resp = chat_completion(
            endpoint, model,
            messages=[{"role": "user", "content": RESPONSE_PROMPT.format(query=query)}],
            max_tokens=60,
            temperature=0.8,
            api_key=api_key,
        )
        # Clean up: remove quotes, strip emoji and non-ASCII
        resp = resp.strip().strip('"').strip("'")
        # Remove emoji and non-printable chars
        resp = ''.join(c for c in resp if c.isascii() and ord(c) >= 32)
        resp = resp.strip()
        # Filter out non-printing or very short responses
        if len(resp) < 1 or len(resp) > 120:
            return None
        # Must contain at least one letter
        if not any(c.isalpha() for c in resp):
            return None
        return resp
    except Exception:
        return None


def generate_diverse_queries(endpoint: str, model: str, count: int = 100,
                            api_key: str = "") -> list[str]:
    """Ask the teacher to generate diverse seed queries."""
    queries = []
    batch_size = 20
    batches = (count + batch_size - 1) // batch_size

    for _ in range(batches):
        try:
            text = chat_completion(
                endpoint, model,
                messages=[{"role": "user", "content": DIVERSIFY_PROMPT}],
                max_tokens=500,
                temperature=0.9,
                api_key=api_key,
            )
            for line in text.split("\n"):
                line = line.strip().strip('"').strip("'").strip("- ").strip("* ")
                # Filter out numbering, empty lines, and lines that look like metadata
                if not line or line.startswith("#"):
                    continue
                if line[0].isdigit() and (". " in line[:4] or ") " in line[:4]):
                    line = line.split(" ", 1)[-1] if " " in line else line
                if len(line) > 5 and len(line) < 80 and any(c.isalpha() for c in line):
                    queries.append(line)
            time.sleep(0.5)  # rate limit
        except Exception as e:
            print(f"  Query generation error: {e}", file=sys.stderr)

    return list(set(queries))  # deduplicate


def distill(
    endpoint: str,
    model: str,
    num_pairs: int = 5000,
    workers: int = 4,
    output: str = "training-data.txt",
    api_key: str = "",
):
    """
    Generate training pairs by:
    1. Starting with SEED_QUERIES
    2. Diversifying via the teacher to get more queries
    3. For each query, generating a natural response
    4. Writing query|response pairs to output file
    """
    print(f"=== Self-Distillation Pipeline ===")
    print(f"  Teacher: {model} @ {endpoint}")
    print(f"  Target:  {num_pairs} training pairs")
    print(f"  Workers: {workers}")
    print()

    # Phase 1: Gather queries
    all_queries = list(SEED_QUERIES)

    # Diversify
    diverse_needed = max(0, num_pairs - len(all_queries))
    if diverse_needed > 0:
        print(f"Generating ~{diverse_needed} diverse queries from teacher...")
        new_queries = generate_diverse_queries(endpoint, model, diverse_needed, api_key)
        all_queries.extend(new_queries)
        print(f"  Got {len(new_queries)} new queries (total: {len(all_queries)})")

    # Deduplicate and limit
    all_queries = list(set(all_queries))[:num_pairs]
    print(f"  Final query set: {len(all_queries)} unique queries")

    # Phase 2: Generate responses in parallel
    print(f"\nGenerating responses ({workers} parallel workers)...")
    pairs = []
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(generate_response, endpoint, model, q, api_key): q
            for q in all_queries
        }

        for future in as_completed(futures):
            query = futures[future]
            try:
                response = future.result()
                if response:
                    pairs.append((query, response))
                else:
                    errors += 1
            except Exception:
                errors += 1

            completed += 1
            if completed % 50 == 0 or completed == len(all_queries):
                print(f"  {completed}/{len(all_queries)} "
                      f"({len(pairs)} valid pairs, {errors} errors)")

    # Phase 3: Write output
    print(f"\nWriting {len(pairs)} pairs to {output}...")
    with open(output, "w") as f:
        for query, response in pairs:
            # Normalize to uppercase (as feedme.py expects)
            q = query.strip().upper()
            r = response.strip().upper()
            f.write(f"{q}|{r}\n")

    print(f"Done! {len(pairs)} training pairs written.")
    print(f"\nNext: train with feedme.py:")
    print(f"  python3 feedme.py --file {output} --epochs 200")

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-distill training data from a teacher model"
    )
    parser.add_argument("--endpoint", "-e",
                        default="http://localhost:8080/v1",
                        help="OpenAI-compatible API endpoint")
    parser.add_argument("--model", "-m",
                        default="gpt-oss-120b",
                        help="Teacher model name")
    parser.add_argument("--pairs", "-n", type=int, default=5000,
                        help="Number of training pairs to generate")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Parallel workers for response generation")
    parser.add_argument("--output", "-o", default="training-data.txt",
                        help="Output file path")
    parser.add_argument("--api-key", "-k", default=None,
                        help="API key (or file path containing key)")
    parser.add_argument("--test", action="store_true",
                        help="Quick test with 10 pairs")
    args = parser.parse_args()

    # Load API key
    api_key = ""
    if args.api_key:
        if os.path.isfile(args.api_key):
            with open(args.api_key) as f:
                api_key = f.read().strip()
        else:
            api_key = args.api_key

    if args.test:
        args.pairs = 10
        args.workers = 1

    distill(
        endpoint=args.endpoint,
        model=args.model,
        num_pairs=args.pairs,
        workers=args.workers,
        output=args.output,
        api_key=api_key,
    )
