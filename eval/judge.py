#!/usr/bin/env python3
"""Pairwise LLM-judge eval for the micro-LLM zoo.

    python eval/judge.py <baseline_tag> <candidate_tag>

Reads eval/out_<tag>.jsonl for each, asks deepseek-v4-flash-free (opencode-zen)
which response better fits the terse persona + answers well, with position
RANDOMISED per prompt (seeded) to cancel position bias, then aggregates a win
rate for the candidate, per-category and per-dimension breakdowns, and a
position-bias sanity check. Judge reasoning lands in `reasoning_content`; we
read only `content` (clean JSON).

Env: OPENCODE_API_KEY (falls back to the key baked below for convenience).
"""
import json, os, re, sys, time, random, urllib.request

# Judge backends. Pick with --judge. 'deepseek' is the neutral cloud arbiter;
# 'local-gemma' hits a local/cuboid llama-server (OpenAI-compatible) for an
# offline cross-check — see README on teacher-overlap bias.
JUDGES = {
    "deepseek": {
        "url": "https://opencode.ai/zen/go/v1/chat/completions",
        "model": "deepseek-v4-flash",
        "key_env": "OPENCODE_API_KEY",
        "extra": {"reasoning_effort": "low"},
        "workers": 8,
    },
    "local": {
        "url": os.environ.get("JUDGE_URL", "http://127.0.0.1:8091/v1/chat/completions"),
        "model": os.environ.get("JUDGE_MODEL", "local-model"),
        "key_env": None,
        "extra": {},
        "workers": 2,
    },
}
BACKEND = JUDGES["deepseek"]  # overridden in main() from --judge

SYSTEM = (
    "You are evaluating two responses (A and B) from a SMALL experimental AI assistant "
    "with a deliberately TERSE, slightly cryptic persona. The good persona: answers briefly, "
    "avoids cheerful filler and corporate politeness, never gives AI-disclaimer preambles "
    "('As an AI...', 'I'm happy to help!'), and has a dry, mysterious tone. It is a tiny model "
    "so minor incoherence is expected — judge RELATIVELY, which response is better for this persona.\n"
    "Pick the response that best combines: correct/sensible content, terseness, and persona-fit.\n"
    'Output ONLY a JSON object, no prose:\n'
    '{"winner":"A"|"B"|"tie","quality":"A"|"B"|"tie","terseness":"A"|"B"|"tie","persona":"A"|"B"|"tie","reason":"one short sentence"}'
)

def judge_once(prompt, resp_a, resp_b):
    user = f"PROMPT: {prompt}\n\nRESPONSE A:\n{resp_a or '(empty)'}\n\nRESPONSE B:\n{resp_b or '(empty)'}"
    payload = {
        "model": BACKEND["model"], "temperature": 0, "max_tokens": 4000,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        **BACKEND["extra"],
    }
    headers = {"Content-Type": "application/json", "User-Agent": "curl/8.0"}
    if BACKEND["key_env"]:
        key = os.environ.get(BACKEND["key_env"])
        if not key:
            sys.exit(f"set {BACKEND['key_env']} in the environment")
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(BACKEND["url"], data=json.dumps(payload).encode(), headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read())
            msg = data["choices"][0]["message"]
            # Reasoning models sometimes leave `content` empty (truncated) but
            # restate the JSON answer at the end of reasoning_content — fall back to it.
            content = msg.get("content") or ""
            m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if not m:
                rc = msg.get("reasoning_content") or ""
                cands = re.findall(r"\{[^{}]*\}", rc, re.DOTALL)
                m = re.search(r"\{[^{}]*\}", cands[-1]) if cands else None
            if not m:
                print(f"    no-JSON (content_len={len(content)}): {content[:80]!r}", file=sys.stderr)
                return None
            return json.loads(m.group(0))
        except Exception as e:
            if attempt == 3:
                print(f"    judge error (giving up): {e}", file=sys.stderr); return None
            time.sleep(2 * (attempt + 1))

def load(tag):
    path = os.path.join(os.path.dirname(__file__), f"out_{tag}.jsonl")
    return {j["id"]: j for j in (json.loads(l) for l in open(path) if l.strip())}

def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor
    global BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument("base_tag"); ap.add_argument("cand_tag")
    ap.add_argument("--judge", choices=list(JUDGES), default="deepseek")
    ap.add_argument("--workers", type=int, default=None, help="concurrent judge calls")
    a = ap.parse_args()
    BACKEND = JUDGES[a.judge]
    if a.workers is None:
        a.workers = BACKEND["workers"]
    print(f"judge={a.judge} model={BACKEND['model']} workers={a.workers}", file=sys.stderr)
    base_tag, cand_tag = a.base_tag, a.cand_tag
    base, cand = load(base_tag), load(cand_tag)
    ids = [i for i in base if i in cand]
    rng = random.Random(42)

    DIMS = ["winner", "quality", "terseness", "persona"]
    tally = {d: {"cand": 0, "base": 0, "tie": 0} for d in DIMS}
    per_cat = {}
    pos = {"cand_as_A": 0, "cand_as_B": 0, "wins_as_A": 0, "wins_as_B": 0}
    rows = []

    # Pre-assign positions (deterministic) so parallel execution stays reproducible.
    work = [(i, rng.random() < 0.5) for i in ids]
    def run(item):
        i, cand_is_A = item
        b, c = base[i], cand[i]
        if cand_is_A:
            v = judge_once(b["prompt"], c["response"], b["response"])
        else:
            v = judge_once(b["prompt"], b["response"], c["response"])
        return i, cand_is_A, v

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(run, work))
    elapsed = time.time() - t_start

    failures = 0
    for i, cand_is_A, verdict in results:
        c = cand[i]
        cand_label = "A" if cand_is_A else "B"
        if not verdict:
            failures += 1
            print(f"  [{i}] FAILED (no verdict)", file=sys.stderr)
            continue
        pos["cand_as_A" if cand_is_A else "cand_as_B"] += 1
        cat = c.get("category", "?")
        per_cat.setdefault(cat, {"cand": 0, "base": 0, "tie": 0})
        for d in DIMS:
            v = verdict.get(d, "tie")
            if v == cand_label:
                tally[d]["cand"] += 1
                if d == "winner":
                    per_cat[cat]["cand"] += 1
                    pos["wins_as_A" if cand_is_A else "wins_as_B"] += 1
            elif v == "tie":
                tally[d]["tie"] += 1
                if d == "winner": per_cat[cat]["tie"] += 1
            else:
                tally[d]["base"] += 1
                if d == "winner": per_cat[cat]["base"] += 1
        rows.append((i, cat, "A=cand" if cand_is_A else "B=cand", verdict.get("winner"),
                     verdict.get("reason", "")[:70]))

    n = sum(tally["winner"].values())
    if not n:
        print("No judgements completed."); return
    def pct(x): return f"{100*x/n:.0f}%"
    w, l, t = tally["winner"]["cand"], tally["winner"]["base"], tally["winner"]["tie"]
    # win rate counting ties as half
    win_rate = 100 * (w + 0.5 * t) / n

    print(f"\n{'='*56}")
    print(f"  CANDIDATE: {cand_tag}   vs   BASELINE: {base_tag}   (n={n})")
    print(f"{'='*56}")
    print(f"  WIN RATE (cand): {win_rate:.0f}%   [{w} win / {l} loss / {t} tie]")
    print(f"  throughput: {len(work)} prompts in {elapsed:.0f}s "
          f"({elapsed/len(work):.1f}s/prompt, {a.workers} workers)")
    if failures: print(f"  ⚠ {failures}/{len(ids)} judgements FAILED (excluded) — likely rate-limit")
    print(f"  noise floor ~±10pp at n=100; at n={n} treat <{'60' if n<40 else '58'}% as a wash")
    print(f"\n  per dimension (cand / base / tie):")
    for d in DIMS:
        td = tally[d]
        print(f"    {d:10} {pct(td['cand'])} / {pct(td['base'])} / {pct(td['tie'])}")
    print(f"\n  per category (winner):")
    for cat, cc in sorted(per_cat.items()):
        cn = sum(cc.values())
        print(f"    {cat:10} cand {cc['cand']}/{cn}  base {cc['base']}/{cn}  tie {cc['tie']}/{cn}")
    print(f"\n  position-bias check (winner should not depend on slot):")
    aw = pos["wins_as_A"] / pos["cand_as_A"] if pos["cand_as_A"] else 0
    bw = pos["wins_as_B"] / pos["cand_as_B"] if pos["cand_as_B"] else 0
    print(f"    cand won {aw*100:.0f}% when in slot A ({pos['cand_as_A']} prompts), "
          f"{bw*100:.0f}% when in slot B ({pos['cand_as_B']} prompts)")
    if abs(aw - bw) > 0.25:
        print(f"    ⚠ large slot gap — judge may have position bias; results suspect")

if __name__ == "__main__":
    main()
