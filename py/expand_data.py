#!/usr/bin/env python3
"""
expand_data.py — Phase 2 data expansion pipeline for the Ghost in the Machine micro-LLM.

Five-phase pipeline:
  Phase 1: Template seed bank (load from file or use built-in set)
  Phase 2: Template expansion via teacher model (~10x more templates)
  Phase 3: Slot filling (word lists → seed prompts)
  Phase 4: Diversity pass (teacher generates ~5 variations per prompt)
  Phase 5: Response generation (length-stratified, teacher-generated)

Resume-safe: saves checkpoint JSON after each phase.

Usage:
  python3 expand_data.py --output training-data-expanded.txt --workers 32
  python3 expand_data.py --phase 3  # run only phase 3
  python3 expand_data.py --phase all --max-prompts 50000
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─── System prompt (from distill.py) ────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a friendly, helpful AI assistant having a natural conversation. "
    "Match the user's tone — casual for casual, brief for brief. "
    "Never ask follow-up questions. Never repeat the user's words back. "
    "Just respond directly and naturally, like a real chat."
)

RESPONSE_USER_TMPL = "Respond to: {query}"

# ─── Length bucket distribution ──────────────────────────────────────────────

LENGTH_BUCKETS = [
    (0.40, "Reply in under 60 characters."),
    (0.35, "Reply in 1-2 sentences."),
    (0.20, "Reply in 2-4 sentences with a little detail."),
    (0.05, "Reply in a short paragraph (3-5 sentences)."),
]

# Precomputed cumulative thresholds for sampling
_BUCKET_CUMULATIVE = []
_cum = 0.0
for _prob, _instr in LENGTH_BUCKETS:
    _cum += _prob
    _BUCKET_CUMULATIVE.append((_cum, _instr))


def sample_length_instruction(
    rng: random.Random | None = None,
    weights: tuple[float, ...] | None = None,
) -> str:
    """Sample a length instruction, optionally overriding the bucket weights.

    weights: 4-tuple of non-negative numbers (terse, short, medium, long).
             Need not sum to 1 — normalised automatically.
             Default: LENGTH_BUCKETS distribution (40/35/20/5).
    """
    if weights is not None:
        if len(weights) != len(LENGTH_BUCKETS):
            raise ValueError(
                f"weights must have {len(LENGTH_BUCKETS)} entries, got {len(weights)}"
            )
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must contain at least one positive value")
        cumulative = []
        cum = 0.0
        for w, (_, instr) in zip(weights, LENGTH_BUCKETS):
            cum += w / total
            cumulative.append((cum, instr))
    else:
        cumulative = _BUCKET_CUMULATIVE

    r = (rng or random).random()
    for threshold, instruction in cumulative:
        if r < threshold:
            return instruction
    return cumulative[-1][1]


# ─── Built-in template seed bank ────────────────────────────────────────────

BUILTIN_TEMPLATES = [
    # Conversation / small talk
    "Tell me something interesting about {topic}",
    "What do you think about {topic}",
    "Do you have any thoughts on {topic}",
    "What's your take on {topic}",
    "How do you feel about {topic}",
    "Have you ever heard of {topic}",
    "What comes to mind when you think of {topic}",
    "Is {topic} something you enjoy",
    "What's the best thing about {topic}",
    "Tell me a fun fact about {topic}",

    # Q&A / facts
    "What is the capital of {country}",
    "What is {country} famous for",
    "Tell me about {country}",
    "What language do they speak in {country}",
    "What's the population of {country}",
    "What's a famous landmark in {country}",
    "What's the weather like in {country}",
    "What should I know before visiting {country}",

    # Jokes
    "Tell me a joke about {topic}",
    "Do you know any funny stories about {topic}",
    "Make me laugh with something about {topic}",

    # Recommendations
    "Recommend a {genre} for {occasion}",
    "What's a good {genre} movie for {occasion}",
    "Can you suggest a {genre} book for {occasion}",
    "What {genre} show should I watch on a {occasion}",
    "What's a great {genre} for a {occasion}",

    # How-to
    "How do I {verb} a {noun}",
    "What's the best way to {verb} a {noun}",
    "Can you walk me through how to {verb} a {noun}",
    "What do I need to {verb} a {noun}",
    "How long does it take to {verb} a {noun}",
    "Is it hard to {verb} a {noun}",
    "What are the steps to {verb} a {noun}",

    # Creative
    "Write a short poem about {topic}",
    "Describe {topic} in a few words",
    "Create a short story involving {topic}",
    "Give me a creative description of {topic}",

    # Opinions
    "Do you prefer {topic} or something else",
    "What would you choose: {topic} or not",
    "Why do people like {topic}",
    "Is {topic} overrated",

    # Goodbyes / greetings
    "Tell me how to say goodbye in a {topic} style",
    "What's a nice way to greet someone interested in {topic}",

    # Meta
    "What would you say to someone who loves {topic}",
    "How would you describe {topic} to a child",
    "If you had to explain {topic} simply, how would you do it",
]


# ─── Slot word lists ─────────────────────────────────────────────────────────

SLOT_WORDS: dict[str, list[str]] = {
    "topic": [
        "dogs", "cats", "space", "music", "art", "cooking", "travel", "books",
        "movies", "nature", "science", "history", "fashion", "sports", "yoga",
        "coffee", "tea", "wine", "cheese", "pasta", "pizza", "sushi", "tacos",
        "hiking", "swimming", "cycling", "running", "dancing", "singing",
        "photography", "painting", "gardening", "baking", "knitting", "gaming",
        "reading", "writing", "meditation", "camping", "fishing", "sailing",
        "chess", "board games", "puzzles", "magic tricks", "origami", "pottery",
        "astronomy", "birds", "butterflies", "trees", "mountains", "oceans",
        "rainforests", "deserts", "islands", "cities", "villages", "markets",
        "sunsets", "rainbows", "thunder", "seasons", "spring", "autumn",
        "friendship", "kindness", "humor", "creativity", "curiosity", "adventure",
        "sleep", "dreams", "time", "memory", "happiness", "gratitude",
        "minimalism", "vintage things", "modern design", "street food",
        "road trips", "languages", "culture", "traditions", "storytelling",
    ],
    "verb": [
        "cook", "fix", "learn", "make", "find", "start", "build", "grow",
        "improve", "organize", "clean", "repair", "create", "plan", "choose",
        "prepare", "manage", "handle", "use", "set up", "decorate", "train",
        "teach", "share", "enjoy", "explore", "discover", "develop", "practise",
        "understand", "design", "write", "paint", "photograph", "record",
        "arrange", "wrap", "assemble", "maintain", "upgrade", "simplify",
        "personalise", "save", "pack", "carry", "transport", "store", "sort",
        "combine", "mix", "heat", "cool", "measure", "adjust", "test",
        "compare", "evaluate", "track", "monitor", "review", "present",
    ],
    "noun": [
        "bike", "garden", "recipe", "app", "dog", "cat", "shelf", "drawer",
        "wardrobe", "kitchen", "bedroom", "balcony", "living room", "backpack",
        "laptop", "camera", "guitar", "piano", "journal", "notebook", "photo",
        "plant", "herb", "flower", "tree", "puzzle", "board game", "book",
        "blanket", "candle", "lamp", "chair", "table", "bag", "hat", "scarf",
        "cake", "soup", "salad", "bread", "smoothie", "sandwich", "stew",
        "website", "video", "podcast", "letter", "card", "gift", "surprise",
        "routine", "habit", "skill", "language", "instrument", "project",
        "space", "community", "team", "friendship", "tradition", "ritual",
    ],
    "country": [
        "France", "Japan", "Brazil", "Italy", "Mexico", "India", "Australia",
        "Canada", "Germany", "Spain", "China", "South Korea", "Argentina",
        "Thailand", "Egypt", "Nigeria", "Kenya", "South Africa", "Sweden",
        "Norway", "Finland", "Denmark", "Netherlands", "Belgium", "Switzerland",
        "Austria", "Portugal", "Greece", "Turkey", "Morocco", "Ethiopia",
        "Tanzania", "Ghana", "Senegal", "Colombia", "Peru", "Chile", "Cuba",
        "Jamaica", "Iceland", "Ireland", "Scotland", "New Zealand", "Singapore",
        "Vietnam", "Cambodia", "Nepal", "Pakistan", "Bangladesh", "Sri Lanka",
    ],
    "genre": [
        "comedy", "drama", "thriller", "romance", "adventure", "mystery",
        "horror", "fantasy", "science fiction", "documentary", "action",
        "animated", "historical", "biographical", "musical", "noir",
        "indie", "foreign", "classic", "contemporary", "feel-good", "dark",
        "family-friendly", "thought-provoking", "suspenseful", "heartwarming",
        "coming-of-age", "road trip", "heist", "spy",
    ],
    "occasion": [
        "birthday", "date night", "rainy day", "long flight", "lazy weekend",
        "girls night", "movie night", "cosy evening", "summer afternoon",
        "winter night", "road trip", "study break", "family gathering",
        "solo evening", "celebration", "cheer-up session", "holiday",
        "new year", "quiet morning", "party", "book club",
    ],
}


# ─── Utility: clean a line of text ───────────────────────────────────────────

PREAMBLE_STARTS = (
    "here are", "sure", "of course", "certainly", "here is",
    "absolutely", "got it", "great", "ok,", "okay,",
    "no problem", "happy to", "i'd be happy", "i'll",
)


def clean_line(line: str) -> str:
    """Strip numbering, bullet prefixes, and preamble lines."""
    line = line.strip().strip('"').strip("'")
    # Strip leading list markers: "1. ", "1) ", "- ", "• ", "* "
    line = line.lstrip("0123456789.-•*) ")
    line = line.strip().strip('"').strip("'")
    return line


def is_preamble(line: str) -> bool:
    """Return True if the line looks like a preamble/meta-comment."""
    low = line.lower()
    return any(low.startswith(p) for p in PREAMBLE_STARTS)


def parse_lines(text: str) -> list[str]:
    """Parse teacher output into a list of usable lines."""
    results = []
    for raw in text.split("\n"):
        line = clean_line(raw)
        if not line:
            continue
        if is_preamble(line):
            continue
        if not any(c.isalpha() for c in line):
            continue
        results.append(line)
    return results


# ─── API calls ───────────────────────────────────────────────────────────────

def chat_completion(
    endpoint: str,
    model: str,
    messages: list,
    max_tokens: int = 200,
    temperature: float = 0.9,
    api_key: str = "",
    system: str = "",
) -> str:
    """Call an OpenAI-compatible chat completions API with retry."""
    url = endpoint.rstrip("/") + "/chat/completions"
    msgs: list = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)
    body = json.dumps({
        "model": model,
        "messages": msgs,
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


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_phases": [], "data": {}}


def save_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


# ─── Phase 1: Template seed bank ─────────────────────────────────────────────

def phase1_templates(templates_file: str | None) -> list[str]:
    """Load templates from file or return built-in set."""
    if templates_file and os.path.exists(templates_file):
        with open(templates_file) as f:
            templates = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(templates)} templates from {templates_file}")
    else:
        if templates_file:
            print(f"  Warning: {templates_file} not found; using built-in templates.")
        templates = list(BUILTIN_TEMPLATES)
        print(f"  Using {len(templates)} built-in templates.")
    return templates


# ─── Phase 2: Template expansion ─────────────────────────────────────────────

def _expand_batch(
    endpoint: str, model: str, api_key: str, examples: list[str], batch_num: int
) -> list[str]:
    """Ask teacher to generate 20 more templates from 5 examples."""
    examples_str = "\n".join(f"  {t}" for t in examples[:5])
    prompt = (
        f"Here are 5 chatbot prompt templates:\n{examples_str}\n\n"
        "Generate 20 more chatbot prompt templates on different topics. "
        "Use {topic}, {verb}, {noun}, {country}, {genre}, or {occasion} as placeholders. "
        "Output one template per line with no numbering or extra text. "
        "No code, math, or programming topics."
    )
    try:
        text = chat_completion(
            endpoint, model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.9,
            api_key=api_key,
        )
        lines = parse_lines(text)
        # Filter: must contain at least one { placeholder or be a coherent sentence
        valid = []
        for line in lines:
            if 10 < len(line) < 120:
                valid.append(line)
        print(f"  Batch {batch_num}: got {len(valid)} templates")
        return valid
    except Exception as e:
        print(f"  Batch {batch_num} error: {e}", file=sys.stderr)
        return []


def phase2_expand(
    templates: list[str],
    endpoint: str,
    model: str,
    api_key: str,
    workers: int,
    target_multiplier: int = 10,
) -> list[str]:
    """Expand templates ~10x using the teacher model."""
    target = len(templates) * target_multiplier
    print(f"  Expanding {len(templates)} templates → target ~{target}")

    # We'll make batches by sampling 5 random templates as examples
    n_batches = max(1, (target - len(templates)) // 15)  # ~15 valid per call
    rng = random.Random(42)

    all_templates = list(templates)
    seen = {t.lower() for t in all_templates}

    batch_args = []
    for i in range(n_batches):
        examples = rng.sample(templates, min(5, len(templates)))
        batch_args.append((endpoint, model, api_key, examples, i + 1))

    with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
        futures = [pool.submit(_expand_batch, *args) for args in batch_args]
        for future in as_completed(futures):
            new_lines = future.result()
            for line in new_lines:
                if line.lower() not in seen:
                    seen.add(line.lower())
                    all_templates.append(line)

    print(f"  Expanded to {len(all_templates)} templates (deduplicated)")
    return all_templates


# ─── Phase 3: Slot filling ────────────────────────────────────────────────────

def fill_slots(template: str, rng: random.Random) -> str:
    """Fill all {slot} placeholders in a template from word lists."""
    import re
    slots = re.findall(r"\{(\w+)\}", template)
    result = template
    for slot in slots:
        words = SLOT_WORDS.get(slot)
        if words:
            result = result.replace("{" + slot + "}", rng.choice(words), 1)
        # Unknown slots: leave as-is
    return result


def phase3_slot_fill(
    templates: list[str],
    samples_per_template: int,
    rng: random.Random,
) -> list[str]:
    """Generate unique filled prompts from templates."""
    all_prompts: list[str] = []
    seen: set[str] = set()

    for template in templates:
        import re
        has_slots = bool(re.search(r"\{\w+\}", template))
        if not has_slots:
            # No slots — use as-is
            key = template.lower()
            if key not in seen:
                seen.add(key)
                all_prompts.append(template)
            continue

        # Try up to 3x samples_per_template to get enough unique ones
        generated = []
        max_attempts = samples_per_template * 3
        for _ in range(max_attempts):
            filled = fill_slots(template, rng)
            key = filled.lower()
            if key not in seen:
                seen.add(key)
                generated.append(filled)
            if len(generated) >= samples_per_template:
                break
        all_prompts.extend(generated)

    print(f"  Generated {len(all_prompts)} unique filled prompts from {len(templates)} templates")
    return all_prompts


# ─── Phase 4: Diversity pass ──────────────────────────────────────────────────

def _diversify_prompt(
    endpoint: str, model: str, api_key: str, prompt: str
) -> list[str]:
    """Generate ~5 natural variations of a prompt."""
    user_msg = (
        f"Given the prompt: '{prompt}', generate 5 different ways a real user might "
        "phrase this. Output one per line, no numbering or extra text."
    )
    try:
        text = chat_completion(
            endpoint, model,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=300,
            temperature=0.9,
            api_key=api_key,
        )
        lines = parse_lines(text)
        valid = [l for l in lines if 5 < len(l) < 200]
        return valid
    except Exception as e:
        print(f"  Diversify error for '{prompt[:30]}...': {e}", file=sys.stderr)
        return []


def phase4_diversity(
    seed_prompts: list[str],
    endpoint: str,
    model: str,
    api_key: str,
    workers: int,
    max_prompts: int,
) -> list[str]:
    """Generate variations for each seed prompt, up to max_prompts total."""
    all_prompts = list(seed_prompts)
    seen = {p.lower() for p in all_prompts}
    total_calls = 0
    total_yield = 0

    # Cap the seed prompts we diversify to avoid exploding past max_prompts
    to_diversify = seed_prompts[:max_prompts]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_diversify_prompt, endpoint, model, api_key, p): p
            for p in to_diversify
        }

        for future in as_completed(futures):
            variations = future.result()
            total_calls += 1
            total_yield += len(variations)

            for v in variations:
                key = v.lower()
                if key not in seen:
                    seen.add(key)
                    all_prompts.append(v)
                    if len(all_prompts) >= max_prompts:
                        break

            if total_calls % 100 == 0:
                print(f"  Diversified {total_calls}/{len(to_diversify)} prompts "
                      f"→ {len(all_prompts)} total")

            if len(all_prompts) >= max_prompts:
                break

    avg_yield = total_yield / max(total_calls, 1)
    print(f"  Diversity pass complete: {len(all_prompts)} prompts "
          f"(avg yield {avg_yield:.1f} per call, {total_calls} calls)")
    return all_prompts[:max_prompts]


# ─── Phase 5: Response generation ────────────────────────────────────────────

def _generate_response(
    endpoint: str,
    model: str,
    api_key: str,
    query: str,
    max_ctx: int,
    rng: random.Random,
    length_weights: tuple[float, ...] | None = None,
) -> tuple[str, str] | None:
    """Generate a response for a query with length-stratified system prompt."""
    length_instr = sample_length_instruction(rng, weights=length_weights)
    system = SYSTEM_PROMPT + " " + length_instr

    try:
        resp = chat_completion(
            endpoint, model,
            messages=[{"role": "user", "content": RESPONSE_USER_TMPL.format(query=query)}],
            max_tokens=300,
            temperature=0.8,
            api_key=api_key,
            system=system,
        )
        # Clean: strip quotes, non-ASCII
        resp = resp.strip().strip('"').strip("'")
        resp = "".join(c for c in resp if c.isascii() and ord(c) >= 32)
        resp = resp.strip()

        if not resp or not any(c.isalpha() for c in resp):
            return None

        # Normalise to uppercase
        q = query.strip().upper()
        r = resp.strip().upper()

        # Length filter: q + separator + r must fit in max_ctx
        if len(q) + len(r) + 1 > max_ctx:
            return None

        return q, r
    except Exception:
        return None


def phase5_responses(
    prompts: list[str],
    endpoint: str,
    model: str,
    api_key: str,
    workers: int,
    max_ctx: int,
    output_file: str,
    checkpoint_path: str,
    state: dict,
    length_weights: tuple[float, ...] | None = None,
) -> list[tuple[str, str]]:
    """Generate responses for all prompts and write to output file."""
    existing_pairs: list[tuple[str, str]] = state["data"].get("phase5_pairs", [])
    seen_queries = {p[0] for p in existing_pairs}

    remaining = [p for p in prompts if p.strip().upper() not in seen_queries]
    print(f"  {len(existing_pairs)} pairs already done; {len(remaining)} prompts remaining")

    pairs = list(existing_pairs)
    completed = 0
    errors = 0
    rng = random.Random(99)  # deterministic per run (reseeded per-call via shuffle)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Build futures with per-call RNG states
        future_map = {}
        for prompt in remaining:
            # Each call gets its own RNG clone to avoid sharing state across threads
            call_rng = random.Random(rng.randint(0, 2**31))
            fut = pool.submit(
                _generate_response, endpoint, model, api_key, prompt, max_ctx,
                call_rng, length_weights
            )
            future_map[fut] = prompt

        for future in as_completed(future_map):
            result = future.result()
            completed += 1

            if result:
                pairs.append(result)
            else:
                errors += 1

            if completed % 200 == 0 or completed == len(remaining):
                print(f"  {completed}/{len(remaining)} "
                      f"({len(pairs)} valid pairs, {errors} filtered/errors)")
                # Save progress checkpoint
                state["data"]["phase5_pairs"] = pairs
                save_checkpoint(checkpoint_path, state)

    print(f"  Response generation done: {len(pairs)} valid pairs, {errors} dropped")
    return pairs


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint
    state = load_checkpoint(checkpoint_path)
    completed = set(state.get("completed_phases", []))

    phases_to_run: set[int]
    if args.phase == "all":
        phases_to_run = {1, 2, 3, 4, 5}
    else:
        phases_to_run = {int(args.phase)}

    rng = random.Random(42)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    if 1 in phases_to_run:
        if 1 in completed:
            templates = state["data"]["templates"]
            print(f"Phase 1: skipped (already done, {len(templates)} templates)")
        else:
            print("Phase 1: Template seed bank")
            templates = phase1_templates(args.templates)
            state["data"]["templates"] = templates
            state["completed_phases"] = list(completed | {1})
            save_checkpoint(checkpoint_path, state)
            completed.add(1)
    else:
        templates = state["data"].get("templates", list(BUILTIN_TEMPLATES))

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    if 2 in phases_to_run:
        if 2 in completed:
            templates = state["data"]["expanded_templates"]
            print(f"Phase 2: skipped (already done, {len(templates)} templates)")
        else:
            print("Phase 2: Template expansion via teacher")
            templates = phase2_expand(
                templates,
                endpoint=args.endpoint,
                model=args.model,
                api_key=args.api_key or "",
                workers=args.workers,
            )
            state["data"]["expanded_templates"] = templates
            state["completed_phases"] = list(completed | {2})
            save_checkpoint(checkpoint_path, state)
            completed.add(2)
    else:
        templates = state["data"].get("expanded_templates", templates)

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    if 3 in phases_to_run:
        if 3 in completed:
            seed_prompts = state["data"]["seed_prompts"]
            print(f"Phase 3: skipped (already done, {len(seed_prompts)} prompts)")
        else:
            print("Phase 3: Slot filling")
            seed_prompts = phase3_slot_fill(
                templates,
                samples_per_template=args.samples_per_template,
                rng=rng,
            )
            state["data"]["seed_prompts"] = seed_prompts
            state["completed_phases"] = list(completed | {3})
            save_checkpoint(checkpoint_path, state)
            completed.add(3)
    else:
        seed_prompts = state["data"].get("seed_prompts", [])

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    if 4 in phases_to_run:
        if 4 in completed:
            all_prompts = state["data"]["all_prompts"]
            print(f"Phase 4: skipped (already done, {len(all_prompts)} prompts)")
        else:
            print("Phase 4: Diversity pass")
            all_prompts = phase4_diversity(
                seed_prompts=seed_prompts,
                endpoint=args.endpoint,
                model=args.model,
                api_key=args.api_key or "",
                workers=args.workers,
                max_prompts=args.max_prompts,
            )
            state["data"]["all_prompts"] = all_prompts
            state["completed_phases"] = list(completed | {4})
            save_checkpoint(checkpoint_path, state)
            completed.add(4)
    else:
        all_prompts = state["data"].get("all_prompts", seed_prompts)

    # ── Phase 5 ──────────────────────────────────────────────────────────────
    if 5 in phases_to_run:
        if 5 in completed:
            print("Phase 5: skipped (already done)")
            pairs = state["data"].get("phase5_pairs", [])
        else:
            print("Phase 5: Response generation")
            pairs = phase5_responses(
                prompts=all_prompts,
                endpoint=args.endpoint,
                model=args.model,
                api_key=args.api_key or "",
                workers=args.workers,
                max_ctx=args.max_ctx,
                output_file=args.output,
                checkpoint_path=checkpoint_path,
                state=state,
                length_weights=args.length_weights,
            )
            state["data"]["phase5_pairs"] = pairs
            state["completed_phases"] = list(completed | {5})
            save_checkpoint(checkpoint_path, state)
            completed.add(5)

        # Write output file
        print(f"\nWriting {len(pairs)} pairs to {args.output}...")
        with open(args.output, "w") as f:
            for q, r in pairs:
                f.write(f"{q}|{r}\n")
        print(f"Done! {len(pairs)} training pairs written.")
        print(f"\nNext: python3 feedme.py --file {args.output} --epochs 40")

    print("\nPipeline complete.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data expansion pipeline for Ghost in the Machine micro-LLM"
    )
    parser.add_argument(
        "--output", "-o", default="training-data-expanded.txt",
        help="Output file for query|response pairs",
    )
    parser.add_argument(
        "--model", "-m", default="gemma4-e4b-distill",
        help="Teacher model name",
    )
    parser.add_argument(
        "--endpoint", "-e", default="http://localhost:8080/v1",
        help="OpenAI-compatible API endpoint",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=32,
        help="Parallel workers for teacher calls",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=50000,
        help="Maximum number of prompts to generate (Phase 4 budget)",
    )
    parser.add_argument(
        "--max-ctx", type=int, default=256,
        help="Max context length for response filtering (use 64 for Wisp, 128 for Shade, 256 for Specter)",
    )
    parser.add_argument(
        "--length-weights", type=str, default=None,
        metavar="T,S,M,L",
        help=(
            "Comma-separated weights for the 4 length buckets: terse,short,medium,long. "
            "Need not sum to 1 (normalised automatically). "
            "Default: 40,35,20,5 (Wisp-friendly). "
            "Specter-friendly: 20,30,35,15. "
            "Example: --length-weights 20,30,35,15"
        ),
    )
    parser.add_argument(
        "--samples-per-template", type=int, default=5,
        help="Unique filled prompts to generate per template (Phase 3)",
    )
    parser.add_argument(
        "--phase", default="all",
        choices=["all", "1", "2", "3", "4", "5"],
        help="Run a specific phase only (default: all)",
    )
    parser.add_argument(
        "--templates", default=None,
        help="Path to file of seed templates (one per line); uses built-in set if omitted",
    )
    parser.add_argument(
        "--checkpoint", default="expand_data_checkpoint.json",
        help="Checkpoint file for resume support",
    )
    parser.add_argument(
        "--api-key", "-k", default=None,
        help="API key or path to file containing one",
    )

    args = parser.parse_args()

    # Parse --length-weights "20,30,35,15" → tuple[float, ...]
    if args.length_weights is not None:
        try:
            parsed = tuple(float(x.strip()) for x in args.length_weights.split(","))
        except ValueError:
            import sys
            print(f"Error: --length-weights must be comma-separated numbers, got: {args.length_weights}", file=sys.stderr)
            sys.exit(1)
        if len(parsed) != 4:
            import sys
            print(f"Error: --length-weights must have exactly 4 values (terse,short,medium,long), got {len(parsed)}", file=sys.stderr)
            sys.exit(1)
        args.length_weights = parsed
    # args.length_weights is now tuple[float,...] | None

    # Resolve API key from file
    if args.api_key and os.path.isfile(args.api_key):
        with open(args.api_key) as f:
            args.api_key = f.read().strip()

    run_pipeline(args)


if __name__ == "__main__":
    main()
