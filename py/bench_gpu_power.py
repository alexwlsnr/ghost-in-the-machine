#!/usr/bin/env python3
"""
GPU power/performance benchmark for Ghost inference.

Measures tokens/second and power draw at the current GPU power limit,
then appends the result to logs/gpu_power_bench.json for comparison
across multiple runs at different power limits.

Usage:
  # Set power limit first (requires sudo):
  #   sudo nvidia-smi -pl 200   # RTX 5080
  #   sudo nvidia-smi -pl 165   # RTX 3070
  #
  # Then run:
  .venv/bin/python3 py/bench_gpu_power.py [--model ckpt/spec512_v2.pt]
"""

import argparse, json, os, subprocess, sys, time, threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_transformer import (
    TinyTransformer, TinyTransformerModern, TinyTransformerTernary,
    generate, encode, PAD_TOKEN, EOS_TOKEN, SEP_TOKEN,
)
import torch

BENCH_PROMPTS = [
    "HELLO HOW ARE YOU",
    "TELL ME A JOKE",
    "WHO ARE YOU",
    "I AM FEELING SAD TODAY",
    "WHAT DO YOU THINK ABOUT MUSIC",
]
TOKENS_PER_RUN = 40
WARMUP_RUNS    = 2
BENCH_RUNS     = 8


def get_gpu_power_watts() -> float:
    """Read current GPU power draw via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw',
             '--format=csv,noheader,nounits'],
            text=True
        ).strip().split('\n')[0]
        return float(out)
    except Exception:
        return 0.0


def get_gpu_power_limit_watts() -> float:
    """Read current enforced power limit."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.limit',
             '--format=csv,noheader,nounits'],
            text=True
        ).strip().split('\n')[0]
        return float(out)
    except Exception:
        return 0.0


def get_gpu_name() -> str:
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            text=True
        ).strip().split('\n')[0].strip()
    except Exception:
        return 'unknown'


def load_model(checkpoint_path: str):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    arch = ck['architecture']
    is_ternary = arch.get('arch') == 'ternary'
    is_modern  = arch.get('arch') == 'modern'

    if is_ternary:
        model = TinyTransformerTernary(
            vocab_size=arch['vocab_size'], d_model=arch['d_model'],
            n_heads=arch['n_heads'], n_layers=arch['n_layers'],
            d_ff=arch['d_ff'], max_len=arch['max_len'],
        )
    elif is_modern:
        state = ck['model_state']
        model = TinyTransformerModern(
            vocab_size=arch['vocab_size'], d_model=arch['d_model'],
            n_heads=arch['n_heads'], n_layers=arch['n_layers'],
            d_ff=arch['d_ff'], max_len=arch['max_len'],
            use_rope='pos_embed.weight' not in state,
            use_swiglu=any('ff.w1' in k for k in state),
            use_rmsnorm=not any('norm1.bias' in k for k in state),
            tie_weights='head.weight' not in state,
        )
    else:
        model = TinyTransformer(
            vocab_size=arch['vocab_size'], d_model=arch['d_model'],
            n_heads=arch['n_heads'], n_layers=arch['n_layers'],
            d_ff=arch['d_ff'], max_len=arch['max_len'],
        )
    model.load_state_dict(ck['model_state'])
    model.eval()
    return model, arch


def run_generation(model, prompt: str, device: str, n_tokens: int) -> int:
    """Greedy generation. Returns number of tokens produced."""
    toks = [*encode(prompt.upper()), SEP_TOKEN]
    toks = toks[:model.max_len - n_tokens - 1]
    generated = 0
    with torch.no_grad():
        for _ in range(n_tokens):
            if len(toks) >= model.max_len - 1:
                break
            x = torch.tensor([toks], dtype=torch.long, device=device)
            logits = model(x)[0, -1]   # (vocab_size,)
            next_tok = int(logits.argmax())
            if next_tok in (EOS_TOKEN, PAD_TOKEN):
                break
            toks.append(next_tok)
            generated += 1
    return generated


def bench_throughput(model, device: str) -> tuple[float, float]:
    """Returns (median_tokens_per_second, avg_power_watts)."""
    model = model.to(device)

    power_samples: list[float] = []
    stop_evt = threading.Event()

    def _poll():
        while not stop_evt.is_set():
            p = get_gpu_power_watts()
            if p > 0:
                power_samples.append(p)
            time.sleep(0.1)

    power_thread = threading.Thread(target=_poll, daemon=True)
    all_tps: list[float] = []

    prompts = (BENCH_PROMPTS * 4)[:WARMUP_RUNS + BENCH_RUNS]
    for i, prompt in enumerate(prompts):
        if i == WARMUP_RUNS:
            power_thread.start()
        t0 = time.perf_counter()
        n = run_generation(model, prompt, device, TOKENS_PER_RUN)
        elapsed = time.perf_counter() - t0
        if i >= WARMUP_RUNS and n > 0:
            all_tps.append(n / elapsed)

    stop_evt.set()
    median_tps = sorted(all_tps)[len(all_tps) // 2] if all_tps else 0.0
    avg_power  = sum(power_samples) / len(power_samples) if power_samples else 0.0
    return median_tps, avg_power


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ckpt/spec512_v2.pt',
                        help='Checkpoint to benchmark')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='logs/gpu_power_bench.json')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    gpu_name  = get_gpu_name()
    power_cap = get_gpu_power_limit_watts()
    print(f"GPU:         {gpu_name}")
    print(f"Power limit: {power_cap:.0f}W")
    print(f"Model:       {args.model}")
    print(f"Warming up ({WARMUP_RUNS} runs) then benchmarking ({BENCH_RUNS} runs)...")

    model, arch = load_model(args.model)
    tps, avg_power = bench_throughput(model, args.device)

    tps_per_watt = tps / avg_power if avg_power > 0 else 0

    result = {
        'gpu':          gpu_name,
        'power_limit_w': power_cap,
        'avg_power_w':   round(avg_power, 1),
        'tok_per_sec':   round(tps, 1),
        'tok_per_watt':  round(tps_per_watt, 3),
        'model':         os.path.basename(args.model),
        'arch':          arch.get('arch', 'classic'),
        'd_model':       arch['d_model'],
        'n_layers':      arch.get('n_layers', '?'),
        'timestamp':     time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    print(f"\n{'─'*50}")
    print(f"  Tokens/sec:       {tps:.1f}")
    print(f"  Avg power draw:   {avg_power:.1f}W")
    print(f"  Efficiency:       {tps_per_watt:.3f} tok/s/W")
    print(f"{'─'*50}")

    # Append to benchmark log
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    results.append(result)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResult appended → {args.output}")

    # Print comparison table if we have multiple results for this GPU
    gpu_results = [r for r in results
                   if r['gpu'] == gpu_name and r['model'] == os.path.basename(args.model)]
    if len(gpu_results) > 1:
        print(f"\nComparison ({gpu_name}, {os.path.basename(args.model)}):")
        print(f"  {'Limit':>7}  {'Draw':>6}  {'tok/s':>7}  {'tok/s/W':>9}  {'Efficiency':>10}")
        gpu_results.sort(key=lambda r: r['power_limit_w'])
        best_tps = max(r['tok_per_sec'] for r in gpu_results)
        for r in gpu_results:
            marker = ' ◄ best perf' if r['tok_per_sec'] == best_tps else ''
            best_eff = max(r2['tok_per_watt'] for r2 in gpu_results)
            eff_marker = ' ◄ best efficiency' if r['tok_per_watt'] == best_eff else ''
            print(f"  {r['power_limit_w']:>6.0f}W  "
                  f"{r['avg_power_w']:>5.0f}W  "
                  f"{r['tok_per_sec']:>7.1f}  "
                  f"{r['tok_per_watt']:>9.3f}  "
                  f"  {int(r['tok_per_sec']/best_tps*100):>3}%{marker}{eff_marker}")


if __name__ == '__main__':
    main()
