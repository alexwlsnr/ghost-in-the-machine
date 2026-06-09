#!/usr/bin/env python3
"""Training throughput benchmark — forward + backward + optimizer step.

Measures batches/sec and tokens/sec at the current GPU power limit.
Use with bench_power_sweep.sh to find the optimal training wattage.

Usage:
  .venv/bin/python3 py/bench_training.py [--arch ternary] [--d-model 512] ...
"""
import argparse, json, os, subprocess, sys, time, threading
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_transformer import TinyTransformer, TinyTransformerTernary

WARMUP_STEPS = 5
BENCH_STEPS  = 20


def get_gpu_power_watts() -> float:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            text=True).strip().split('\n')[0]
        return float(out)
    except Exception:
        return 0.0


def get_gpu_power_limit_watts() -> float:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits'],
            text=True).strip().split('\n')[0]
        return float(out)
    except Exception:
        return 0.0


def get_gpu_name() -> str:
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            text=True).strip().split('\n')[0].strip()
    except Exception:
        return 'unknown'


def make_model(args):
    if args.arch == 'ternary':
        return TinyTransformerTernary(
            vocab_size=args.vocab_size, d_model=args.d_model,
            n_heads=args.n_heads, n_layers=args.n_layers,
            d_ff=args.d_ff, max_len=args.max_len,
        )
    return TinyTransformer(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, max_len=args.max_len,
    )


def bench(model, batch_size, max_len, vocab_size, device, amp):
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler() if amp else None

    # Synthetic data — same shape as real training
    x = torch.randint(0, vocab_size, (batch_size, max_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, max_len), device=device)

    power_samples: list[float] = []
    stop_evt = threading.Event()

    def _poll():
        while not stop_evt.is_set():
            p = get_gpu_power_watts()
            if p > 0:
                power_samples.append(p)
            time.sleep(0.05)

    # Warmup
    for _ in range(WARMUP_STEPS):
        opt.zero_grad()
        if amp:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            opt.step()
    torch.cuda.synchronize()

    # Benchmark
    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    t0 = time.perf_counter()

    for _ in range(BENCH_STEPS):
        opt.zero_grad()
        if amp:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            opt.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop_evt.set()

    batches_per_sec = BENCH_STEPS / elapsed
    tokens_per_sec  = BENCH_STEPS * batch_size * max_len / elapsed
    avg_power       = sum(power_samples) / len(power_samples) if power_samples else 0.0

    return batches_per_sec, tokens_per_sec, avg_power


def main():
    parser = argparse.ArgumentParser()
    # Architecture — defaults match Ternary Shade target
    parser.add_argument('--arch',       default='ternary', choices=['classic', 'ternary'])
    parser.add_argument('--d-model',    type=int, default=512)
    parser.add_argument('--n-heads',    type=int, default=8)
    parser.add_argument('--n-layers',   type=int, default=6)
    parser.add_argument('--d-ff',       type=int, default=2048)
    parser.add_argument('--max-len',    type=int, default=256)
    parser.add_argument('--vocab-size', type=int, default=258)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--amp',        action='store_true', default=True)
    parser.add_argument('--no-amp',     dest='amp', action='store_false')
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--output',     default='logs/gpu_power_bench.json')
    args = parser.parse_args()

    gpu_name  = get_gpu_name()
    power_cap = get_gpu_power_limit_watts()
    n_params  = sum(p.numel() for p in make_model(args).parameters())

    print(f"GPU:         {gpu_name}")
    print(f"Power limit: {power_cap:.0f}W")
    print(f"Arch:        {args.arch}  d={args.d_model} L={args.n_layers} ff={args.d_ff}")
    print(f"Params:      {n_params:,}  batch={args.batch_size}  ctx={args.max_len}  amp={args.amp}")
    print(f"Warming up ({WARMUP_STEPS} steps) then benchmarking ({BENCH_STEPS} steps)...")

    model = make_model(args)
    bps, tps, avg_power = bench(model, args.batch_size, args.max_len,
                                args.vocab_size, args.device, args.amp)
    tps_per_watt = tps / avg_power if avg_power > 0 else 0

    print(f"\n  Batches/sec:      {bps:.1f}")
    print(f"  Tokens/sec:       {tps/1000:.1f}K")
    print(f"  Avg power draw:   {avg_power:.1f}W")
    print(f"  Efficiency:       {tps_per_watt:.0f} tok/s/W")

    result = {
        'gpu': gpu_name, 'power_limit_w': power_cap, 'avg_power_w': round(avg_power, 1),
        'batches_per_sec': round(bps, 2), 'tok_per_sec': round(tps, 0),
        'tok_per_watt': round(tps_per_watt, 1),
        'arch': args.arch, 'd_model': args.d_model, 'n_layers': args.n_layers,
        'batch_size': args.batch_size, 'max_len': args.max_len, 'amp': args.amp,
        'mode': 'training',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            try: results = json.load(f)
            except Exception: pass
    results.append(result)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResult appended → {args.output}")


if __name__ == '__main__':
    main()
