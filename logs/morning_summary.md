# Overnight Training Results — Morning Summary

## Val Loss Leaderboard (all models)

| Model | Val Loss | Architecture | Data | Notes |
|-------|----------|--------------|------|-------|
| Spec512 v1.1 | **0.519** (still running) | Classic | 135K multi-turn | New all-time best |
| shade_modern_arch | **0.580** | Modern (RMSNorm+SwiGLU+RoPE+tying) | 29K clean | Best Shade |
| shade_compact | **0.589** | Modern (compact d_ff=1024) | 29K clean | Smaller & nearly as good |
| shade_clean (baseline) | 0.596 | Classic | 29K clean | Previous best |
| wisp_shade_data | **0.613** | Modern (no RoPE) | 29K clean | Best Wisp! |
| shade_winner_large | 0.607 | Modern | 37K large | More data ≠ better |
| wisp_best_v2 | 0.629 | Modern (no RoPE, lr=0.001) | 29K clean | LR=0.003 better |

## Key Findings

### 1. Modern Architecture Genuinely Helps Shade
RMSNorm + SwiGLU + RoPE + weight tying: 0.580 vs 0.596 baseline (+2.7%)
All four components together. Best config confirmed.

### 2. Response Loss Masking HURTS on Short Pairs
shade_modern_all (arch+masking): 0.642 — WORSE than arch-only (0.580)
On ~50-char avg utterances, masking removes ~50% of training signal.
May help for longer multi-turn sequences (untested).

### 3. More Data ≠ Better Quality (data quality matters more)
shade_winner_large (37K noisy): 0.607
shade_modern_arch (29K clean): 0.580
Clean curated data beats more noisy data.

### 4. Bigger Model Needs More Data
shade_plus (d=512, 18M params on 29K): 0.668 — worse than 10.9M model
Larger capacity requires proportionally more data to generalize.

### 5. Wisp: Data > Architecture
wisp_modern (modern arch, 2.6K data): 1.084 — terrible
wisp_shade_data (modern arch no-RoPE, 29K data): 0.613 — great!
Modern architecture requires more data to use its capacity.
Higher LR (0.003) better than 0.001 for Wisp.

### 6. RoPE Less Important at ctx=128
No-rope Wisp on 29K data: 0.613 (same as with RoPE)
At short contexts, learned positional embeddings work fine.

## New Models Deployed (feature/modern-arch)
- **Shade Modern** (8bit, 14.4MB): val_loss 0.580 — live in browser
- **Shade Compact** (8bit, 11MB): val_loss 0.589 — live in browser  
- **Wisp Modern** (8bit, 4.6MB): val_loss 0.613 — live in browser

## Browser Inference: What Was Implemented
Full modern arch support in TS/Wasm:
- rms_norm_f32 Wasm kernel (no mean-centering)
- silu_f32 Wasm kernel (for SwiGLU activation)
- mul_vec_f32 Wasm kernel (gate*value element-wise)
- RoPE in TS: precomputed cos/sin, consecutive-pair rotation
  Key bug found+fixed: PyTorch uses (x[0],x[1]),(x[2],x[3]) pairs
  not (x[0],x[half]),(x[1],x[half+1]) split pairs
- SwiGLU: gate+val pre-allocated outside layer loop

## Still Running
- Spec512 v1.1: ep=46, best=0.519 at ep=45, patience ~19/20 remaining

## Next Steps (when user wakes)
1. Serialize Spec512 v1.1 when it finishes (~1-2 more hours)
2. Retrain Wisp with LR=0.003 + modern arch on 29K (should beat 0.613)
3. Consider running Spec512 v1.1 config (135K data, modern arch) for Shade
4. Merge feature/modern-arch to main
5. Plan Revenant ternary model
