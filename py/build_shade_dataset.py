#!/usr/bin/env python3
"""
Build training dataset for Shade BPE Ternary (22MB / ~11M params).

Downloads and processes open-source conversational datasets, converts to
Q|R pipe-separated uppercase format compatible with train_transformer.py.

Target: ~500K high-quality lines → ~25M BPE tokens.
This lays groundwork for Revenant (larger model, same pipeline).

Sources (all freely accessible, commercial-friendly or non-commercial research):
  - HuggingFaceH4/ultrachat_200k  (MIT)
  - HuggingFaceTB/smoltalk        (Apache 2.0, built for small models)
  - OpenAssistant/oasst2          (Apache 2.0, en-filtered)
  - daily_dialog                  (CC BY-NC-SA 4.0)
  - allenai/prosocial-dialog      (CC BY 4.0)
  - google/Synthetic-Persona-Chat (CC BY 4.0)
  - facebook/empathetic_dialogues (CC BY-NC 4.0)

Usage:
  .venv/bin/python3 py/build_shade_dataset.py [--out data/shade_bpe_train.txt]
"""
import argparse, os, re, random, sys, time
from pathlib import Path

os.environ['HF_DATASETS_CACHE'] = '.hf_cache'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean(text: str) -> str:
    """Preserve case, collapse whitespace, strip control chars.
    Shade BPE Ternary drops the all-caps restriction — BPE vocab handles mixed case natively.
    """
    text = text.strip()
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def is_good(turn: str, min_len=8, max_len=400) -> bool:
    """Basic quality filter."""
    if not turn or len(turn) < min_len or len(turn) > max_len:
        return False
    # Too many non-ASCII chars (URLs, code, foreign text)
    ascii_ratio = sum(1 for c in turn if ord(c) < 128) / len(turn)
    if ascii_ratio < 0.85:
        return False
    # Reject if it's mostly a URL or code block
    tupper = turn.upper()
    if 'HTTP' in tupper or 'WWW.' in tupper or '```' in turn:
        return False
    return True


def pair_to_line(q: str, r: str) -> str | None:
    q, r = clean(q), clean(r)
    if is_good(q) and is_good(r):
        return f"{q}|{r}"
    return None


def multiturn_to_lines(turns: list[str]) -> list[str]:
    """Convert a list of alternating utterances to Q|R|Q|R... lines.
    Yields both single-turn and multi-turn slices for variety.
    """
    lines = []
    # Single-turn pairs
    for i in range(0, len(turns) - 1, 2):
        line = pair_to_line(turns[i], turns[i + 1])
        if line:
            lines.append(line)
    # Multi-turn slices (up to 4 turns)
    if len(turns) >= 4:
        for i in range(0, len(turns) - 3, 2):
            parts = [clean(t) for t in turns[i:i + 4]]
            if all(is_good(p) for p in parts):
                lines.append('|'.join(parts))
    return lines


# ─── Dataset processors ────────────────────────────────────────────────────

def process_ultrachat(max_items=80_000) -> list[str]:
    log("Loading UltraChat 200K...")
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft',
                      trust_remote_code=False)
    lines, count = [], 0
    for item in ds:
        if count >= max_items:
            break
        msgs = item.get('messages', [])
        turns = [m['content'] for m in msgs if m['role'] in ('user', 'assistant')]
        lines.extend(multiturn_to_lines(turns))
        count += 1
        if count % 10000 == 0:
            log(f"  UltraChat: {count}/{max_items} items → {len(lines)} lines")
    log(f"  UltraChat done: {len(lines)} lines")
    return lines


def process_smoltalk(max_items=60_000) -> list[str]:
    log("Loading SmolTalk...")
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train',
                      trust_remote_code=False)
    lines, count = [], 0
    for item in ds:
        if count >= max_items:
            break
        msgs = item.get('messages', [])
        turns = [m['content'] for m in msgs if m['role'] in ('user', 'assistant')]
        lines.extend(multiturn_to_lines(turns))
        count += 1
        if count % 10000 == 0:
            log(f"  SmolTalk: {count}/{max_items} items → {len(lines)} lines")
    log(f"  SmolTalk done: {len(lines)} lines")
    return lines


def process_oasst2(max_items=30_000) -> list[str]:
    log("Loading OASST2 (English only)...")
    from datasets import load_dataset
    ds = load_dataset('OpenAssistant/oasst2', split='train',
                      trust_remote_code=False)
    # Build message tree: id → message
    msgs = {r['message_id']: r for r in ds if r.get('lang') == 'en'}

    # Find root→leaf paths for conversation reconstruction
    children: dict = {}
    for mid, m in msgs.items():
        parent = m.get('parent_id')
        if parent and parent in msgs:
            children.setdefault(parent, []).append(mid)

    def extract_paths(mid, path):
        path = path + [mid]
        kids = children.get(mid, [])
        if not kids:
            yield path
        else:
            for kid in kids[:2]:  # limit branching
                yield from extract_paths(kid, path)

    roots = [mid for mid, m in msgs.items() if not m.get('parent_id') or m.get('parent_id') not in msgs]
    lines, count = [], 0
    for root in roots:
        if count >= max_items:
            break
        for path in extract_paths(root, []):
            turns = [msgs[mid]['text'] for mid in path if mid in msgs]
            lines.extend(multiturn_to_lines(turns))
            count += 1
    log(f"  OASST2 done: {len(lines)} lines")
    return lines


def process_daily_dialog() -> list[str]:
    log("Loading DailyDialog...")
    from datasets import load_dataset
    ds = load_dataset('daily_dialog', split='train', trust_remote_code=False)
    lines = []
    for item in ds:
        turns = item.get('dialog', [])
        lines.extend(multiturn_to_lines(turns))
    log(f"  DailyDialog done: {len(lines)} lines")
    return lines


def process_prosocial(max_items=50_000) -> list[str]:
    log("Loading ProSocial Dialog...")
    from datasets import load_dataset
    ds = load_dataset('allenai/prosocial-dialog', split='train',
                      trust_remote_code=False)
    lines, count = [], 0
    for item in ds:
        if count >= max_items:
            break
        ctx = item.get('context', [])
        resp = item.get('response', '')
        if ctx and resp:
            # Last context turn is human, response is assistant
            q = ctx[-1] if ctx else ''
            line = pair_to_line(q, resp)
            if line:
                lines.append(line)
        count += 1
    log(f"  ProSocial done: {len(lines)} lines")
    return lines


def process_synthetic_persona_chat() -> list[str]:
    log("Loading Synthetic-Persona-Chat...")
    from datasets import load_dataset
    ds = load_dataset('google/Synthetic-Persona-Chat', split='train',
                      trust_remote_code=False)
    lines = []
    for item in ds:
        conv = item.get('Best Generated Conversation', item.get('conversation', ''))
        if not conv:
            continue
        # Format is "User: ...\nBot: ..." or similar
        parts = re.split(r'\n(?:User\s*\d*|Bot\s*\d*|Person\s*\d*|Human|Assistant):\s*', conv, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        lines.extend(multiturn_to_lines(parts))
    log(f"  Synthetic-Persona-Chat done: {len(lines)} lines")
    return lines


def process_empathetic_dialogues(max_items=20_000) -> list[str]:
    log("Loading Empathetic Dialogues...")
    from datasets import load_dataset
    ds = load_dataset('facebook/empathetic_dialogues', split='train',
                      trust_remote_code=False)
    # Group by conv_id
    convs: dict = {}
    for item in ds:
        cid = item['conv_id']
        convs.setdefault(cid, []).append((item['utterance_idx'], item['utterance']))
    lines, count = [], 0
    for cid, utts in convs.items():
        if count >= max_items:
            break
        utts.sort(key=lambda x: x[0])
        turns = [u for _, u in utts]
        lines.extend(multiturn_to_lines(turns))
        count += 1
    log(f"  Empathetic Dialogues done: {len(lines)} lines")
    return lines


def load_existing(paths: list[str]) -> list[str]:
    lines = []
    for p in paths:
        if Path(p).exists():
            raw = Path(p).read_text(errors='replace').splitlines()
            lines.extend(l.strip() for l in raw if l.strip() and '|' in l)
            log(f"  Existing {p}: {len(raw)} lines")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/shade_bpe_train.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Dataset names to skip (e.g. smoltalk ultrachat)')
    args = parser.parse_args()

    random.seed(args.seed)
    skip = set(args.skip)
    all_lines = []

    # 1. Existing high-quality data first
    log("Loading existing datasets...")
    existing = load_existing([
        'data/spec512_v12_clean.txt',
        'data/scenarios_2turn.txt',
        'data/scenarios_3turn.txt',
        'data/scenarios_multiturn.txt',
        'data/scenarios.txt',
    ])
    all_lines.extend(existing)
    log(f"Existing data: {len(existing)} lines")

    # 2. Open source datasets
    processors = [
        ('ultrachat',      process_ultrachat),
        ('smoltalk',       process_smoltalk),
        ('oasst2',         process_oasst2),
        ('daily_dialog',   process_daily_dialog),
        ('prosocial',      process_prosocial),
        ('persona_chat',   process_synthetic_persona_chat),
        ('empathetic',     process_empathetic_dialogues),
    ]

    for name, fn in processors:
        if name in skip:
            log(f"Skipping {name}")
            continue
        try:
            lines = fn()
            all_lines.extend(lines)
            log(f"Running total: {len(all_lines):,} lines")
        except Exception as e:
            log(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()

    # 3. Deduplicate
    log(f"Deduplicating {len(all_lines):,} lines...")
    seen = set()
    deduped = []
    for line in all_lines:
        key = line[:80]  # cheap dedup on first 80 chars
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    log(f"After dedup: {len(deduped):,} lines")

    # 4. Shuffle
    random.shuffle(deduped)

    # 5. Write
    Path(args.out).write_text('\n'.join(deduped) + '\n')
    log(f"Wrote {args.out}: {len(deduped):,} lines")

    # Stats
    log("\n=== Dataset Summary ===")
    log(f"Total lines:     {len(deduped):,}")
    avg_len = sum(len(l) for l in deduped) / max(len(deduped), 1)
    log(f"Avg line length: {avg_len:.0f} chars")
    est_bpe_tokens = int(len(deduped) * avg_len / 3.5)
    log(f"Est BPE tokens:  {est_bpe_tokens/1_000_000:.1f}M (at ~3.5 chars/token)")
    log(f"Chinchilla fit:  {'OK for ~' + str(int(est_bpe_tokens/20/1_000_000)) + 'M param model' if est_bpe_tokens > 0 else '?'}")


if __name__ == '__main__':
    main()
