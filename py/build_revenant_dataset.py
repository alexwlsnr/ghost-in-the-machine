#!/usr/bin/env python3
"""
Build training datasets for Shade ablations, Revenant, and Spectre v2.

Blend modes (--blend flag):
  baseline    — same sources as shade_bpe_train.txt, reproduced cleanly
  factual     — baseline + TriviaQA + NaturalQuestions (factual grounding test)
  quality     — baseline sources but aggressively filtered, no extra data
  scale       — full UltraChat + SmolTalk (not sampled), ~3x more data
  domain      — 50% Ghost scenarios + persona-heavy, less general chat
  memory      — baseline + PersonaChat + MSC (multi-turn context retention test)
  revenant    — everything, max scale, for full Revenant pretraining
  spectre_v2  — max scale + SODA + 6-turn sequences, tuned for ctx=512

Usage:
  .venv/bin/python3 py/build_revenant_dataset.py --blend factual --out data/ablation_factual.txt
  .venv/bin/python3 py/build_revenant_dataset.py --blend memory  --out data/ablation_memory.txt
  .venv/bin/python3 py/build_revenant_dataset.py --blend revenant --out data/revenant_train.txt
  .venv/bin/python3 py/build_revenant_dataset.py --blend spectre_v2 --out data/spectre_v2_train.txt
"""
import argparse, os, re, random, sys, time
from pathlib import Path

os.environ['HF_DATASETS_CACHE'] = '.hf_cache'

STAGE_DIR_RE = re.compile(
    r'\((?:SMIL(?:ES|ING)|LAUGH(?:S|ING)|SIGH(?:S|ING)|NOD(?:S|DING)|'
    r'FROWN(?:S|ING)|CRY(?:ING)?|WINK(?:S|ING)|GASP(?:S|ING)|'
    r'WHISPER(?:S|ING)|SHRUG(?:S|GING)|PAUSE(?:S|D)?|CHUCKL(?:ES|ING)|'
    r'GROAN(?:S|ING)|CLEARS? THROAT|LOOKS? (?:AT|AWAY|DOWN|UP))\)',
    re.IGNORECASE
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def is_good(turn: str, min_len=10, max_len=500, strict=False) -> bool:
    if not turn or len(turn) < min_len or len(turn) > max_len:
        return False
    ascii_ratio = sum(1 for c in turn if ord(c) < 128) / len(turn)
    if ascii_ratio < 0.85:
        return False
    tupper = turn.upper()
    # Reject URLs, code blocks, technical artifacts
    if any(x in tupper for x in ('HTTP', 'WWW.', '```', 'SUDO ', 'NOPASSWD', 'ALL=(ALL)')):
        return False
    # Reject stage directions (the (SMILING) problem from SODA data)
    if STAGE_DIR_RE.search(turn):
        return False
    # Reject all-caps short turns (likely formatting artifacts)
    if len(turn) < 30 and turn == turn.upper() and any(c.isalpha() for c in turn):
        return False
    if strict:
        # Stricter: reject short responses, repetition, exclamation spam
        if len(turn) < 20:
            return False
        words = turn.lower().split()
        if len(words) >= 4:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:  # >50% repeated words
                return False
        if turn.count('!') > 3 or turn.count('?') > 3:
            return False
    return True


def pair_to_line(q: str, r: str, strict=False) -> str | None:
    q, r = clean(q), clean(r)
    if is_good(q, strict=strict) and is_good(r, strict=strict):
        return f"{q}|{r}"
    return None


def multiturn_to_lines(turns: list[str], strict=False, max_turns=4) -> list[str]:
    lines = []
    for i in range(0, len(turns) - 1, 2):
        line = pair_to_line(turns[i], turns[i + 1], strict=strict)
        if line:
            lines.append(line)
    # 4-turn sequences
    if len(turns) >= 4:
        for i in range(0, len(turns) - 3, 2):
            parts = [clean(t) for t in turns[i:i + 4]]
            if all(is_good(p, strict=strict) for p in parts):
                lines.append('|'.join(parts))
    # 6-turn sequences (for longer-context sources / ctx=512)
    if max_turns >= 6 and len(turns) >= 6:
        for i in range(0, len(turns) - 5, 2):
            parts = [clean(t) for t in turns[i:i + 6]]
            if all(is_good(p, strict=strict) for p in parts):
                lines.append('|'.join(parts))
    return lines


# ─── Existing sources (from build_shade_dataset.py) ────────────────────────

def process_ultrachat(max_items=80_000, strict=False) -> list[str]:
    log(f"Loading UltraChat (max={max_items:,})...")
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft', trust_remote_code=False)
    lines, count = [], 0
    for item in ds:
        if count >= max_items:
            break
        msgs = item.get('messages', [])
        turns = [m['content'] for m in msgs if m['role'] in ('user', 'assistant')]
        lines.extend(multiturn_to_lines(turns, strict=strict))
        count += 1
        if count % 20000 == 0:
            log(f"  UltraChat: {count:,}/{max_items:,} → {len(lines):,} lines")
    log(f"  UltraChat done: {len(lines):,} lines")
    return lines


def process_smoltalk(max_items=60_000, strict=False) -> list[str]:
    log(f"Loading SmolTalk (max={max_items:,})...")
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceTB/smoltalk', 'all', split='train', trust_remote_code=False)
    lines, count = [], 0
    for item in ds:
        if count >= max_items:
            break
        msgs = item.get('messages', [])
        turns = [m['content'] for m in msgs if m['role'] in ('user', 'assistant')]
        lines.extend(multiturn_to_lines(turns, strict=strict))
        count += 1
        if count % 20000 == 0:
            log(f"  SmolTalk: {count:,}/{max_items:,} → {len(lines):,} lines")
    log(f"  SmolTalk done: {len(lines):,} lines")
    return lines


def process_oasst2(max_items=30_000, strict=False) -> list[str]:
    log("Loading OASST2 (English)...")
    from datasets import load_dataset
    ds = load_dataset('OpenAssistant/oasst2', split='train', trust_remote_code=False)
    msgs = {r['message_id']: r for r in ds if r.get('lang') == 'en'}
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
            for kid in kids[:2]:
                yield from extract_paths(kid, path)

    roots = [mid for mid, m in msgs.items()
             if not m.get('parent_id') or m.get('parent_id') not in msgs]
    lines, count = [], 0
    for root in roots:
        if count >= max_items:
            break
        for path in extract_paths(root, []):
            turns = [msgs[mid]['text'] for mid in path if mid in msgs]
            lines.extend(multiturn_to_lines(turns, strict=strict))
            count += 1
    log(f"  OASST2 done: {len(lines):,} lines")
    return lines


# Sources that failed to load (HF script-deprecation etc.) are recorded here so the
# build can surface them loudly instead of silently shipping a dataset with holes.
FAILED_SOURCES: list[str] = []


def process_daily_dialog(strict=False) -> list[str]:
    log("Loading DailyDialog...")
    from datasets import load_dataset
    try:
        # ConvLab mirror, streamed: the canonical `daily_dialog` is script-based (HF
        # dropped script support), and ConvLab's non-streaming build hits an arrow
        # generation error on a bad record — streaming iterates past it cleanly.
        ds = load_dataset('ConvLab/dailydialog', split='train',
                          trust_remote_code=False, streaming=True)
        lines = []
        for item in ds:
            turns = [t['utterance'] for t in item.get('turns', []) if t.get('utterance')]
            lines.extend(multiturn_to_lines(turns, strict=strict))
        log(f"  DailyDialog done: {len(lines):,} lines")
        return lines
    except Exception as e:
        log(f"  DailyDialog failed ({e}), skipping")
        FAILED_SOURCES.append('DailyDialog')
        return []


def process_prosocial(max_items=50_000, strict=False) -> list[str]:
    log("Loading ProSocial Dialog...")
    from datasets import load_dataset
    ds = load_dataset('allenai/prosocial-dialog', split='train', trust_remote_code=False)
    lines, count = [], 0
    for item in ds:
        if count >= max_items:
            break
        ctx = item.get('context', '')   # plain string, not a list
        resp = item.get('response', '')
        if ctx and resp:
            line = pair_to_line(ctx, resp, strict=strict)
            if line:
                lines.append(line)
        count += 1
    log(f"  ProSocial done: {len(lines):,} lines")
    return lines


def process_synthetic_persona_chat(strict=False) -> list[str]:
    log("Loading Synthetic-Persona-Chat...")
    from datasets import load_dataset
    ds = load_dataset('google/Synthetic-Persona-Chat', split='train', trust_remote_code=False)
    lines = []
    for item in ds:
        conv = item.get('Best Generated Conversation', item.get('conversation', ''))
        if not conv:
            continue
        parts = re.split(r'\n(?:User\s*\d*|Bot\s*\d*|Person\s*\d*|Human|Assistant):\s*',
                         conv, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        lines.extend(multiturn_to_lines(parts, strict=strict))
    log(f"  Synthetic-Persona-Chat done: {len(lines):,} lines")
    return lines


def process_empathetic_dialogues(max_items=20_000, strict=False) -> list[str]:
    log("Loading Empathetic Dialogues...")
    from datasets import load_dataset
    try:
        # Estwld mirror: parquet, pre-grouped into role-tagged conversation turns
        # (facebook/empathetic_dialogues is script-based and no longer loads).
        ds = load_dataset('Estwld/empathetic_dialogues_llm', split='train', trust_remote_code=False)
        lines, count = [], 0
        for item in ds:
            if count >= max_items:
                break
            turns = [t['content'] for t in item.get('conversations', []) if t.get('content')]
            lines.extend(multiturn_to_lines(turns, strict=strict))
            count += 1
        log(f"  Empathetic Dialogues done: {len(lines):,} lines")
        return lines
    except Exception as e:
        log(f"  Empathetic Dialogues failed ({e}), skipping")
        FAILED_SOURCES.append('Empathetic Dialogues')
        return []


def process_soda(max_dialogues=500_000, strict=False) -> list[str]:
    """SODA (Allen AI) — 1.49M social dialogues, CC-BY 4.0.
    Average utterance ~16 chars; ideal fit for our conversational format.
    Extracts 2-, 4-, and 6-turn sequences to maximise ctx=512 utilisation."""
    log(f"Loading SODA (max_dialogues={max_dialogues:,})...")
    from datasets import load_dataset
    try:
        ds = load_dataset('allenai/soda', split='train', trust_remote_code=False)
        lines, count = [], 0
        for item in ds:
            if count >= max_dialogues:
                break
            dialogue = item.get('dialogue', [])
            if dialogue:
                lines.extend(multiturn_to_lines(dialogue, strict=strict, max_turns=6))
            count += 1
            if count % 100_000 == 0:
                log(f"  SODA: {count:,}/{max_dialogues:,} → {len(lines):,} lines")
        log(f"  SODA done: {len(lines):,} lines from {count:,} dialogues")
        return lines
    except Exception as e:
        log(f"  SODA failed ({e}), skipping")
        FAILED_SOURCES.append('SODA')
        return []


# ─── New factual sources ────────────────────────────────────────────────────

def process_triviaqa(max_items=40_000) -> list[str]:
    log("Loading TriviaQA...")
    from datasets import load_dataset
    ds = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='train',
                      trust_remote_code=False)
    lines, count = [], 0
    for item in ds:
        if count >= max_items:
            break
        q = item.get('question', '').strip()
        answers = item.get('answer', {})
        # Use the normalised answer (short, canonical)
        ans = answers.get('value', '') or (answers.get('aliases', [''])[0] if answers.get('aliases') else '')
        ans = ans.strip()
        if q and ans and len(q) > 10 and len(ans) > 1:
            line = pair_to_line(q, ans)
            if line:
                lines.append(line)
        count += 1
    log(f"  TriviaQA done: {len(lines):,} lines")
    return lines


def process_natural_questions(max_items=30_000) -> list[str]:
    log("Loading Natural Questions (short answers)...")
    from datasets import load_dataset
    # simplified NQ — easier to process than full NQ
    try:
        ds = load_dataset('sentence-transformers/natural-questions', split='train',
                          trust_remote_code=False)
        lines, count = [], 0
        for item in ds:
            if count >= max_items:
                break
            q = item.get('query', item.get('question', '')).strip()
            ans = item.get('answer', item.get('positive', '')).strip()
            if isinstance(ans, list):
                ans = ans[0] if ans else ''
            if q and ans and len(q) > 10 and 5 < len(ans) < 300:
                line = pair_to_line(q, ans)
                if line:
                    lines.append(line)
            count += 1
        log(f"  NaturalQuestions done: {len(lines):,} lines")
        return lines
    except Exception as e:
        log(f"  NaturalQuestions failed ({e}), trying fallback...")
        return []


def process_eli5(max_items=25_000, strict=False) -> list[str]:
    """ELI5 — explain-like-I'm-5 answers, good accessible factual prose."""
    log("Loading ELI5...")
    from datasets import load_dataset
    try:
        # sentence-transformers mirror: parquet, already flattened to question/answer
        # pairs (eli5_category is script-based and no longer loads).
        ds = load_dataset('sentence-transformers/eli5', split='train', trust_remote_code=False)
        lines, count = [], 0
        for item in ds:
            if count >= max_items:
                break
            q = item.get('question', '').strip()
            ans = item.get('answer', '').strip()
            # ELI5 answers can be long — truncate to 400 chars at sentence boundary
            if len(ans) > 400:
                ans = ans[:400].rsplit('.', 1)[0] + '.'
            line = pair_to_line(q, ans, strict=strict)
            if line:
                lines.append(line)
            count += 1
        log(f"  ELI5 done: {len(lines):,} lines")
        return lines
    except Exception as e:
        log(f"  ELI5 failed ({e}), skipping")
        FAILED_SOURCES.append('ELI5')
        return []


def process_sciq(strict=False) -> list[str]:
    """SciQ — science multiple-choice Q&A, converts to Q|correct-answer pairs."""
    log("Loading SciQ...")
    from datasets import load_dataset
    try:
        ds = load_dataset('allenai/sciq', split='train', trust_remote_code=False)
        lines = []
        for item in ds:
            q = item.get('question', '').strip()
            ans = item.get('correct_answer', '').strip()
            support = item.get('support', '').strip()
            # Prefer the support text (explains why) over bare answer
            resp = support if support and len(support) > len(ans) else ans
            if q and resp:
                line = pair_to_line(q, resp, strict=strict)
                if line:
                    lines.append(line)
        log(f"  SciQ done: {len(lines):,} lines")
        return lines
    except Exception as e:
        log(f"  SciQ failed ({e}), skipping")
        FAILED_SOURCES.append('SciQ')
        return []


# ─── Memory / context-retention sources ────────────────────────────────────

def process_personachat(max_convs=10_000, strict=False) -> list[str]:
    """PersonaChat — casual 2-person dialogues grounded in persona facts.
    Each speaker has 4-5 facts ('I have three dogs', 'I work as a teacher')
    woven naturally into conversation. Trains implicit fact-referencing."""
    log(f"Loading PersonaChat (max_convs={max_convs:,})...")
    from datasets import load_dataset
    try:
        # AlekseyKorshuk mirror: parquet; each row is one conversation with a list of
        # `utterances`, each carrying the running `history` + `candidates` (gold last).
        # (personachat_truecased is script-based and no longer loads.)
        ds = load_dataset('AlekseyKorshuk/persona-chat', split='train', trust_remote_code=False)
        lines, count = [], 0
        for item in ds:
            if count >= max_convs:
                break
            utts = item.get('utterances', [])
            if not utts:
                continue
            last = utts[-1]                                   # accumulates the full dialogue
            full_conv = list(last.get('history', [])) + [last['candidates'][-1]]
            lines.extend(multiturn_to_lines(full_conv, strict=strict))
            count += 1
        log(f"  PersonaChat done: {len(lines):,} lines from {count:,} convs")
        return lines
    except Exception as e:
        log(f"  PersonaChat failed ({e}), skipping")
        FAILED_SOURCES.append('PersonaChat')
        return []


def process_msc(max_sessions=15_000, strict=False) -> list[str]:
    """Multi-Session Chat — conversations across 4 sequential sessions.
    Facts stated in session 0 accumulate into persona for session 1+.
    Specifically designed to address 'goldfish memory' in open-domain chat."""
    log(f"Loading Multi-Session Chat (max_sessions={max_sessions:,})...")
    from datasets import load_dataset
    try:
        ds = load_dataset('nayohan/multi_session_chat', split='train',
                          trust_remote_code=False)
        lines, count = [], 0
        for item in ds:
            if count >= max_sessions:
                break
            dialogue = item.get('dialogue', [])
            if dialogue:
                lines.extend(multiturn_to_lines(dialogue, strict=strict))
            count += 1
        log(f"  MSC done: {len(lines):,} lines from {count:,} sessions")
        return lines
    except Exception as e:
        log(f"  MSC failed ({e}), skipping")
        FAILED_SOURCES.append('MSC')
        return []


# ─── Ghost / domain sources ─────────────────────────────────────────────────

def load_ghost_scenarios(weight=3) -> list[str]:
    """Load Ghost scenario files, repeated `weight` times to increase domain signal."""
    scenario_files = [
        'data/scenarios_2turn.txt',
        'data/scenarios_3turn.txt',
        'data/scenarios_multiturn.txt',
        'data/scenarios.txt',
    ]
    lines = []
    for p in scenario_files:
        if Path(p).exists():
            raw = [l.strip() for l in Path(p).read_text(errors='replace').splitlines()
                   if l.strip() and '|' in l]
            lines.extend(raw)
            log(f"  Ghost {p}: {len(raw):,} lines")
    log(f"  Ghost total: {len(lines):,} lines  (will repeat ×{weight})")
    return lines * weight


def load_existing_clean(paths: list[str], strict=False) -> list[str]:
    lines = []
    for p in paths:
        if Path(p).exists():
            raw = [l.strip() for l in Path(p).read_text(errors='replace').splitlines()
                   if l.strip() and '|' in l]
            if strict:
                # Re-filter existing data through stricter rules
                parts_list = [l.split('|') for l in raw]
                raw = ['|'.join(parts) for parts in parts_list
                       if all(is_good(p, strict=True) for p in parts)]
            lines.extend(raw)
            log(f"  Loaded {p}: {len(raw):,} lines")
    return lines


# ─── Blend definitions ──────────────────────────────────────────────────────

def build_baseline(args) -> list[str]:
    lines = load_existing_clean([
        'data/spec512_v12_clean.txt',
        'data/scenarios_2turn.txt', 'data/scenarios_3turn.txt',
        'data/scenarios_multiturn.txt', 'data/scenarios.txt',
    ])
    lines += process_ultrachat(80_000)
    lines += process_smoltalk(60_000)
    lines += process_oasst2(30_000)
    lines += process_daily_dialog()
    lines += process_prosocial(50_000)
    lines += process_synthetic_persona_chat()
    lines += process_empathetic_dialogues(20_000)
    return lines


def build_factual(args) -> list[str]:
    lines = build_baseline(args)
    lines += process_triviaqa(40_000)
    lines += process_natural_questions(30_000)
    lines += process_eli5(25_000)
    lines += process_sciq()
    return lines


def build_quality(args) -> list[str]:
    """Same sources as baseline, but strict filtering throughout."""
    strict = True
    lines = load_existing_clean([
        'data/spec512_v12_clean.txt',
        'data/scenarios_2turn.txt', 'data/scenarios_3turn.txt',
        'data/scenarios_multiturn.txt', 'data/scenarios.txt',
    ], strict=strict)
    lines += process_ultrachat(80_000, strict=strict)
    lines += process_smoltalk(60_000, strict=strict)
    lines += process_oasst2(30_000, strict=strict)
    lines += process_daily_dialog(strict=strict)
    lines += process_prosocial(50_000, strict=strict)
    lines += process_synthetic_persona_chat(strict=strict)
    lines += process_empathetic_dialogues(20_000, strict=strict)
    return lines


def build_scale(args) -> list[str]:
    """Full UltraChat + SmolTalk, not sampled — roughly 3× baseline data."""
    lines = load_existing_clean([
        'data/spec512_v12_clean.txt',
        'data/scenarios_2turn.txt', 'data/scenarios_3turn.txt',
        'data/scenarios_multiturn.txt', 'data/scenarios.txt',
    ])
    lines += process_ultrachat(200_000)   # full dataset
    lines += process_smoltalk(200_000)    # full dataset
    lines += process_oasst2(30_000)
    lines += process_daily_dialog()
    lines += process_prosocial(50_000)
    lines += process_synthetic_persona_chat()
    lines += process_empathetic_dialogues(20_000)
    return lines


def build_domain(args) -> list[str]:
    """50% Ghost scenarios (repeated), rest from persona/empathetic sources."""
    ghost = load_ghost_scenarios(weight=5)  # heavy repetition
    lines = ghost
    lines += load_existing_clean(['data/spec512_v12_clean.txt'])
    lines += process_synthetic_persona_chat()
    lines += process_empathetic_dialogues(20_000)
    lines += process_prosocial(30_000)
    lines += process_daily_dialog()
    return lines


def build_memory(args) -> list[str]:
    """Baseline + PersonaChat + MSC — tests whether context retention improves
    when the model sees persona-grounded and cross-session dialogues."""
    lines = build_baseline(args)
    # PersonaChat and MSC repeated ×2 to upweight the memory signal
    lines += process_personachat(10_000) * 2
    lines += process_msc(15_000) * 2
    return lines


def build_revenant(args) -> list[str]:
    """Everything — max scale for Revenant pretraining target ~1M+ lines."""
    lines = load_ghost_scenarios(weight=3)
    lines += load_existing_clean(['data/spec512_v12_clean.txt'])
    lines += process_ultrachat(200_000)
    lines += process_smoltalk(200_000)
    lines += process_oasst2(50_000)
    lines += process_daily_dialog()
    lines += process_prosocial(50_000)
    lines += process_synthetic_persona_chat()
    lines += process_empathetic_dialogues(30_000)
    lines += process_triviaqa(40_000)
    lines += process_natural_questions(30_000)
    lines += process_eli5(25_000)
    lines += process_sciq()
    lines += process_personachat(10_000)
    lines += process_msc(15_000)
    return lines


def build_spectre_v2(args) -> list[str]:
    """Max-scale blend for Spectre v2 (ctx=512).
    Adds SODA (500K dialogues) and uses 6-turn sequences throughout
    to maximise token utilisation for the wider context window."""
    lines = load_ghost_scenarios(weight=5)
    lines += load_existing_clean(['data/spec512_v12_clean.txt'])
    # Primary conversational sources — full caps
    lines += process_soda(500_000)
    lines += process_ultrachat(200_000)
    lines += process_smoltalk(200_000)
    lines += process_oasst2(60_000)
    lines += process_daily_dialog()
    lines += process_prosocial(50_000)
    lines += process_synthetic_persona_chat()
    lines += process_empathetic_dialogues(30_000)
    # Factual grounding
    lines += process_triviaqa(40_000)
    lines += process_natural_questions(30_000)
    lines += process_eli5(25_000)
    lines += process_sciq()
    # Memory / persona (upweighted ×2)
    pc = process_personachat(15_000)
    lines += pc * 2
    msc = process_msc(20_000)
    lines += msc * 2
    return lines


BLENDS = {
    'baseline':   build_baseline,
    'factual':    build_factual,
    'quality':    build_quality,
    'scale':      build_scale,
    'domain':     build_domain,
    'memory':     build_memory,
    'revenant':   build_revenant,
    'spectre_v2': build_spectre_v2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', choices=list(BLENDS), default='baseline',
                        help='Which data blend to build')
    parser.add_argument('--out', default=None,
                        help='Output path (defaults to data/ablation_<blend>.txt)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-failed-sources', action='store_true',
                        help='exit 0 even if some data sources failed to load '
                             '(default: exit non-zero so wrappers notice the holes)')
    args = parser.parse_args()

    if args.out is None:
        default_names = {'revenant': 'data/revenant_train.txt',
                         'spectre_v2': 'data/spectre_v2_train.txt'}
        args.out = default_names.get(args.blend, f'data/ablation_{args.blend}.txt')

    log(f"Building blend: {args.blend} → {args.out}")
    random.seed(args.seed)

    all_lines = BLENDS[args.blend](args)

    # Deduplicate
    log(f"Deduplicating {len(all_lines):,} lines...")
    seen = set()
    deduped = []
    for line in all_lines:
        key = line[:80]
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    log(f"After dedup: {len(deduped):,} lines")

    random.shuffle(deduped)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(deduped) + '\n')

    log(f"\n=== {args.blend.upper()} Dataset Summary ===")
    log(f"Lines:           {len(deduped):,}")
    avg_len = sum(len(l) for l in deduped) / max(len(deduped), 1)
    log(f"Avg line length: {avg_len:.0f} chars")
    est_tokens = int(len(deduped) * avg_len / 3.5)
    log(f"Est BPE tokens:  {est_tokens / 1_000_000:.1f}M")
    chinchilla_fit = est_tokens // 20
    log(f"Chinchilla fit:  ~{chinchilla_fit/1_000_000:.0f}M param model")
    log(f"Output:          {args.out}")

    # Loud surfacing: a source that failed to load yields [] — indistinguishable from a
    # genuine empty source unless we flag it. (This is how the v2 build silently shipped
    # without DailyDialog/Empathetic/ELI5/PersonaChat.)
    if FAILED_SOURCES:
        log("\n" + "!" * 60)
        log(f"WARNING: {len(FAILED_SOURCES)} data source(s) FAILED to load and were dropped:")
        for s in FAILED_SOURCES:
            log(f"  - {s}")
        log("The dataset above is INCOMPLETE. Investigate before training on it.")
        log("!" * 60)
        if not args.allow_failed_sources:
            sys.exit(1)


if __name__ == '__main__':
    main()
