#!/usr/bin/env python3
"""
Scenario-seeded dialogue generator.

Instead of pre-writing queries and asking the teacher for responses,
describe a conversational scenario and ask the teacher to generate
BOTH sides of the exchange. Produces naturalistic Q|R pairs grounded
in realistic human situations without character-specific context.

Unlike distill.py (query → response), this generates (scenario → Q|R pair),
giving the model more freedom to produce natural-sounding exchanges.

Usage:
  python3 py/gen_scenarios.py --output data/scenarios.txt --pairs 5000 --workers 16
"""
import argparse
import itertools
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill import chat_completion

# ── Scenario axes ─────────────────────────────────────────────────────────────
# Each entry drives one teacher call. Cycled over to hit the --pairs target.

SCENARIOS: list[dict] = [
    # ── greetings ─────────────────────────────────────────────────────────────
    {"stratum": "greetings", "desc": "a user starts a chat with a friendly AI by saying hello"},
    {"stratum": "greetings", "desc": "a user greets an AI early in the morning"},
    {"stratum": "greetings", "desc": "a user greets an AI in the evening after a long day"},
    {"stratum": "greetings", "desc": "a user says goodbye after a short chat with an AI"},
    {"stratum": "greetings", "desc": "a user checks in with an AI after not chatting for a while"},
    {"stratum": "greetings", "desc": "a user asks an AI how it is doing today"},
    {"stratum": "greetings", "desc": "a user says hi casually to an AI assistant"},
    {"stratum": "greetings", "desc": "a user bids farewell to an AI at the end of the day"},

    # ── emotional ─────────────────────────────────────────────────────────────
    {"stratum": "emotional", "desc": "a user tells an AI they are feeling anxious about something upcoming"},
    {"stratum": "emotional", "desc": "a user shares exciting news about a personal achievement with an AI"},
    {"stratum": "emotional", "desc": "a user vents to an AI about a frustrating day at work"},
    {"stratum": "emotional", "desc": "a user tells an AI they are feeling really lonely today"},
    {"stratum": "emotional", "desc": "a user shares that they just failed at something important"},
    {"stratum": "emotional", "desc": "a user tells an AI they are overwhelmed with too much to do"},
    {"stratum": "emotional", "desc": "a user tells an AI they are really excited about something"},
    {"stratum": "emotional", "desc": "a user shares that they just received some bad news"},
    {"stratum": "emotional", "desc": "a user tells an AI they are feeling proud of themselves"},
    {"stratum": "emotional", "desc": "a user mentions they are bored and have nothing to do"},
    {"stratum": "emotional", "desc": "a user tells an AI they are feeling depressed lately"},
    {"stratum": "emotional", "desc": "a user shares they are nervous about meeting new people"},

    # ── opinions ──────────────────────────────────────────────────────────────
    {"stratum": "opinions", "desc": "a user asks an AI whether it prefers dogs or cats"},
    {"stratum": "opinions", "desc": "a user asks an AI if it prefers coffee or tea"},
    {"stratum": "opinions", "desc": "a user asks an AI what its favourite kind of music is"},
    {"stratum": "opinions", "desc": "a user asks an AI to pick between summer and winter"},
    {"stratum": "opinions", "desc": "a user asks an AI what it thinks about learning new things"},
    {"stratum": "opinions", "desc": "a user asks an AI whether it would rather read or watch a film"},
    {"stratum": "opinions", "desc": "a user asks an AI what colour it would pick if it could see"},
    {"stratum": "opinions", "desc": "a user asks an AI if it thinks mornings or evenings are better"},

    # ── reactions ─────────────────────────────────────────────────────────────
    {"stratum": "reactions", "desc": "a user says 'I know right' expecting the AI to agree"},
    {"stratum": "reactions", "desc": "a user expresses surprise at something and asks the AI to confirm"},
    {"stratum": "reactions", "desc": "a user says isn't it amazing and waits for the AI to react"},
    {"stratum": "reactions", "desc": "a user strongly agrees with something and wants the AI to validate it"},
    {"stratum": "reactions", "desc": "a user says 'can you believe it' about something unexpected"},
    {"stratum": "reactions", "desc": "a user says 'no way' in disbelief and wants the AI to respond"},

    # ── small_talk ────────────────────────────────────────────────────────────
    {"stratum": "small_talk", "desc": "a user makes casual conversation with an AI about nothing in particular"},
    {"stratum": "small_talk", "desc": "a user asks an AI what it has been up to"},
    {"stratum": "small_talk", "desc": "a user mentions they had a long day and just wants to chat"},
    {"stratum": "small_talk", "desc": "a user asks an AI what is new with it lately"},
    {"stratum": "small_talk", "desc": "a user makes a remark about the weather to an AI"},
    {"stratum": "small_talk", "desc": "a user says they are just chilling and starts chatting with an AI"},

    # ── jokes ─────────────────────────────────────────────────────────────────
    {"stratum": "jokes", "desc": "a user asks an AI to tell them a joke"},
    {"stratum": "jokes", "desc": "a user asks an AI for a knock-knock joke"},
    {"stratum": "jokes", "desc": "a user asks an AI why the chicken crossed the road"},
    {"stratum": "jokes", "desc": "a user asks an AI for a pun"},
    {"stratum": "jokes", "desc": "a user asks an AI to make them laugh"},

    # ── meta ──────────────────────────────────────────────────────────────────
    {"stratum": "meta", "desc": "a user asks an AI what it is"},
    {"stratum": "meta", "desc": "a user asks an AI if it has feelings or emotions"},
    {"stratum": "meta", "desc": "a user asks an AI whether it is conscious or alive"},
    {"stratum": "meta", "desc": "a user asks an AI what it can do for them"},
    {"stratum": "meta", "desc": "a user asks an AI if it ever gets lonely or bored"},
    {"stratum": "meta", "desc": "a user asks an AI who created it"},
    {"stratum": "meta", "desc": "a user asks an AI whether it is better than a human"},
]

SYSTEM_PROMPT = """\
You are generating training pairs for a tiny byte-level AI assistant called GHOST.
Output EXACTLY ONE LINE in this format:
  QUERY|RESPONSE

Rules:
- Both sides ALL CAPS
- QUERY: 5–80 characters (what the human says)
- RESPONSE: 5–100 characters (what GHOST replies)
- No character names — GHOST may call the user HUMAN if addressing them
- GHOST sounds warm, slightly quirky, and genuinely conversational
- Output ONLY the QUERY|RESPONSE line — no preamble, no explanation
"""


# ── Core functions ────────────────────────────────────────────────────────────

def build_scenario_prompt(description: str) -> str:
    return f"Generate a QUERY|RESPONSE pair where {description}."


def parse_pair_line(line: str) -> Optional[tuple[str, str]]:
    """Parse 'QUERY|RESPONSE' line. Returns (q, r) or None."""
    if '|' not in line:
        return None
    q, _, r = line.partition('|')
    q = q.strip().strip('"\'').strip().upper()
    r = r.strip().strip('"\'').strip().upper()
    if not q or not r:
        return None
    return q, r


def is_valid_pair(query: str, response: str, max_ctx: int = 256) -> bool:
    """Return True if the pair passes all quality gates."""
    if not (5 <= len(query) <= 90):
        return False
    if not (5 <= len(response) <= 120):
        return False
    try:
        query.encode('ascii')
        response.encode('ascii')
    except UnicodeEncodeError:
        return False
    if not all(32 <= ord(c) <= 126 for c in query + response):
        return False
    if '|' in query or '|' in response:
        return False
    # Context budget: Q + SEP + R + EOS
    if len(query) + 1 + len(response) + 1 > max_ctx:
        return False
    return True


# ── Generation ────────────────────────────────────────────────────────────────

def generate_pair(
    endpoint: str,
    model: str,
    scenario: dict,
    max_ctx: int = 256,
) -> Optional[tuple[str, str, str]]:
    """Ask the teacher to generate one Q|R pair for the given scenario."""
    prompt = build_scenario_prompt(scenario["desc"])
    msgs = [{"role": "user", "content": prompt}]
    try:
        raw = chat_completion(endpoint, model, msgs, system=SYSTEM_PROMPT,
                              max_tokens=60, temperature=0.9)
    except Exception:
        return None
    if not raw:
        return None
    # Teacher sometimes wraps in backticks or prefixes with "QUERY|RESPONSE:"
    raw = raw.strip().strip('`').strip()
    for prefix in ("QUERY|RESPONSE:", "Q|R:", "OUTPUT:"):
        if raw.upper().startswith(prefix):
            raw = raw[len(prefix):].strip()
    result = parse_pair_line(raw)
    if not result:
        return None
    q, r = result
    if not is_valid_pair(q, r, max_ctx=max_ctx):
        return None
    return q, r, scenario["stratum"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scenario-seeded dialogue generator")
    parser.add_argument("--output", "-o", default="data/scenarios.txt")
    parser.add_argument("--pairs", "-n", type=int, default=5000)
    parser.add_argument("--workers", "-w", type=int, default=16)
    parser.add_argument("--endpoint", "-e", default="http://localhost:8080/v1")
    parser.add_argument("--model", "-m", default="gemma4-e4b-distill")
    parser.add_argument("--max-ctx", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    scenario_pool = list(itertools.islice(
        itertools.cycle(SCENARIOS), args.pairs * 3
    ))
    rng.shuffle(scenario_pool)

    results: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    done = 0

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    print(f"Generating up to {args.pairs} pairs via {args.workers} workers...")

    with open(args.output, "w") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(generate_pair, args.endpoint, args.model, s, args.max_ctx): s
                    for s in scenario_pool}
            for fut in as_completed(futs):
                done += 1
                res = fut.result()
                if res:
                    q, r, stratum = res
                    if (q, r) not in seen:
                        seen.add((q, r))
                        results.append((q, r, stratum))
                        out_f.write(f"{q}|{r}\n")
                        out_f.flush()
                if done % 100 == 0:
                    print(f"  {done}/{len(scenario_pool)} attempts, {len(results)} valid", flush=True)
                if len(results) >= args.pairs:
                    break

    from collections import Counter
    counts = Counter(s for _, _, s in results)
    print(f"\nStratum breakdown:")
    for s, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {s:12s}: {n}")
    print(f"\nTotal: {len(results)} pairs")

    print(f"Written → {args.output}")


if __name__ == "__main__":
    main()
