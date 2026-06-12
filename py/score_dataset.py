#!/usr/bin/env python3
"""Persona-conditional dataset scorer (terseness + comprehension) via llama-server.

For each training line (single-turn `Q|R` or multi-turn `Q1|R1|Q2|R2|...`), a small
instruct judge rates how well the ASSISTANT replies fit the terse-ghost persona on a
1-5 scale (5 = short, direct, on-topic, correct; 1 = rambling / verbose / off-topic /
nonsensical). Output is constrained to a single digit. Scores are written to a JSONL
sidecar (`{"i": <line_index>, "score": <1-5>}`) that the v3 dataset builder filters on.

Why llama-server (not llama-cpp-python): its OpenAI endpoint does continuous batching,
so N concurrent requests are batched server-side — far higher throughput for 2.5M pairs.

Usage (attach to a running server):
    llama-server -m JUDGE.gguf -ngl 99 -c 2048 -np 24 --host 127.0.0.1 --port 8090
    python3 py/score_dataset.py --data data/spectre_v2_train.txt \
        --out logs/scores_spectre_v2.jsonl --url http://127.0.0.1:8090 --concurrency 24

Or let it launch/teardown the server itself:
    python3 py/score_dataset.py --data data/spectre_v2_train.txt \
        --out logs/scores_spectre_v2.jsonl \
        --model ~/.lmstudio/models/.../Qwen3.5-0.8B-Q8_0.gguf --launch --concurrency 24
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

SYSTEM = (
    "You evaluate training examples for a TERSE assistant called Ghost that gives "
    "short, direct, on-topic replies (one or two sentences, no rambling, no filler). "
    "Rate how well the ASSISTANT turns below fit that target on a 1-5 scale:\n"
    "5 = ideal: short, direct, correct, on-topic\n"
    "3 = acceptable but a bit long or generic\n"
    "1 = poor: verbose/rambling, off-topic, or nonsensical\n"
    "Reply with ONLY a single digit 1-5. No words, no punctuation."
)
MAX_TRANSCRIPT_CHARS = 1600  # keep prompt well under judge ctx


def line_to_transcript(line: str) -> str:
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if len(parts) < 2:
        return ""
    rows = []
    for i, p in enumerate(parts):
        rows.append(f"{'USER' if i % 2 == 0 else 'ASSISTANT'}: {p}")
    t = "\n".join(rows)
    return t[:MAX_TRANSCRIPT_CHARS]


def score_request(url: str, model: str, transcript: str, timeout: float) -> int | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": transcript},
        ],
        "temperature": 0.0,
        "max_tokens": 2,
        "cache_prompt": True,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
        text = body["choices"][0]["message"]["content"]
        for ch in text:
            if ch in "12345":
                return int(ch)
        return None
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None


def wait_for_health(url: str, timeout_s: float = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") in ("ok", "no slot available"):
                    return True
        except Exception:
            time.sleep(2)
    return False


def launch_server(model: str, host: str, port: int, slots: int, ctx: int):
    cmd = [
        "llama-server", "-m", os.path.expanduser(model),
        "-ngl", "99", "-c", str(ctx), "-np", str(slots),
        "--host", host, "--port", str(port),
    ]
    print(f"[score] launching: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="JSONL sidecar of {i, score}")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="judge", help="model name for the API / path if --launch")
    ap.add_argument("--launch", action="store_true", help="launch llama-server from --model gguf")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0, help="score at most N unscored lines (0=all)")
    args = ap.parse_args()

    lines = open(args.data, encoding="utf-8", errors="replace").read().splitlines()
    print(f"[score] {len(lines):,} lines in {args.data}")

    done: set[int] = set()
    if os.path.exists(args.out):
        for ln in open(args.out):
            try:
                done.add(json.loads(ln)["i"])
            except Exception:
                pass
        print(f"[score] resuming — {len(done):,} already scored")

    todo = [i for i in range(len(lines)) if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[score] scoring {len(todo):,} lines with concurrency {args.concurrency}")

    proc = None
    if args.launch:
        proc = launch_server(args.model, "127.0.0.1", int(args.url.rsplit(":", 1)[1]),
                             args.concurrency, args.ctx)
        if not wait_for_health(args.url):
            print("[score] server did not become healthy", file=sys.stderr)
            if proc:
                proc.terminate()
            sys.exit(1)
        print("[score] server healthy")

    out_f = open(args.out, "a")
    lock = Lock()
    counter = {"n": 0, "ok": 0, "t0": time.time()}

    def work(i: int):
        transcript = line_to_transcript(lines[i])
        score = None if not transcript else score_request(
            args.url, args.model, transcript, args.timeout)
        with lock:
            out_f.write(json.dumps({"i": i, "score": score}) + "\n")
            counter["n"] += 1
            if score is not None:
                counter["ok"] += 1
            if counter["n"] % 500 == 0:
                el = time.time() - counter["t0"]
                rate = counter["n"] / el
                eta = (len(todo) - counter["n"]) / rate / 3600
                out_f.flush()
                print(f"[score] {counter['n']:,}/{len(todo):,} "
                      f"({counter['ok']} scored) {rate:.0f}/s ETA {eta:.1f}h", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(work, todo))
    finally:
        out_f.flush()
        out_f.close()
        if proc:
            proc.terminate()

    el = time.time() - counter["t0"]
    print(f"[score] done — {counter['n']:,} lines, {counter['ok']:,} scored, {el/3600:.2f}h")


if __name__ == "__main__":
    main()
