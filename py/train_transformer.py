#!/usr/bin/env python3
"""
Byte-level Tiny Transformer training for Tier 2.5 "Ghost Transformer"

Tokenization: 256 byte values + PAD=256
Training: autoregressive on concatenated query+response pairs
Inference: prompt → generate until PAD token
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from contextlib import nullcontext
from typing import List, Optional, Tuple

# ─── Constants ─────────────────────────────────────────────────────

VOCAB_SIZE = 258  # bytes 0-255 + PAD_TOKEN (256) + EOS_TOKEN (257)
PAD_TOKEN = 256
EOS_TOKEN = 257
SEP_TOKEN = 1    # ASCII SOH — query/response separator. Injected between Q and R
                 # in training so the model learns a clean response zone.
                 # At inference, injected after the prompt so generate() outputs
                 # pure response bytes (never the separator itself).
DEFAULT_MAX_LEN = 64


# ─── Model ──────────────────────────────────────────────────────────

class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = DEFAULT_MAX_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) int64 token ids. Returns (batch, seq_len, vocab_size)."""
        B, T = x.shape
        device = x.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        tok_emb = self.token_embed(x) * math.sqrt(self.d_model)
        pos_emb = self.pos_embed(positions)
        h = tok_emb + pos_emb

        # Create causal + padding mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        pad_mask = (x == PAD_TOKEN)

        h = self.encoder(
            h,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
            is_causal=True,
        )
        h = self.ln_final(h)
        return self.head(h)

    def compute_quantization_loss(self, weight_bits: int = 4) -> torch.Tensor:
        """Push weights toward the actual quantization grid used at inference.

        Uses the same scale as serialize.py (absmax / max_int) so QAT targets
        the same 16 levels (4-bit) or 256 levels (8-bit) used at inference.
        The old formula (95th-percentile scale, round to nearest 1) was
        accidentally ternary-like and misaligned with the serializer.
        """
        max_int = float(2 ** (weight_bits - 1) - 1)  # 7.0 for 4-bit, 127.0 for 8-bit
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            if p.dim() <= 1:
                continue
            with torch.no_grad():
                absmax = p.abs().max().clamp(min=1e-8)
                scale = absmax / max_int
                target = torch.clamp(torch.round(p / scale), -max_int - 1, max_int)
            loss = loss + F.mse_loss(p / scale, target)
        return loss

    def get_quantized_params(self, weight_bits: int = 4) -> dict:
        """Extract quantized parameters for Wasm serialization."""
        max_val = 2 ** (weight_bits - 1) - 1
        min_val = -(2 ** (weight_bits - 1))
        params = {}

        def quant_weight(w):
            scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
            return torch.clamp(torch.round(w / scale), min_val, max_val) \
                .detach().cpu().numpy().astype(np.int8)

        def quant_bias(b):
            return torch.round(b * 32).detach().cpu().numpy().astype(np.int16)

        def quant_ln(t):
            """Quantize layer norm params to i16 with scale 256."""
            return torch.round(t * 256).clamp(-32768, 32767) \
                .detach().cpu().numpy().astype(np.int16)

        def float16_param(t):
            return t.detach().cpu().numpy().astype(np.float16)

        # Embeddings
        params['token_embed'] = quant_weight(self.token_embed.weight)
        params['pos_embed'] = quant_weight(self.pos_embed.weight)

        # Encoder layers
        for li, layer in enumerate(self.encoder.layers):
            pfx = f'enc{li}'
            d = self.d_model

            # Q, K, V are stored as a single weight/bias in PyTorch's MultiheadAttention
            in_w = layer.self_attn.in_proj_weight
            in_b = layer.self_attn.in_proj_bias
            params[f'{pfx}_q_weight'] = quant_weight(in_w[:d])
            params[f'{pfx}_k_weight'] = quant_weight(in_w[d:2*d])
            params[f'{pfx}_v_weight'] = quant_weight(in_w[2*d:])
            params[f'{pfx}_q_bias'] = quant_bias(in_b[:d])
            params[f'{pfx}_k_bias'] = quant_bias(in_b[d:2*d])
            params[f'{pfx}_v_bias'] = quant_bias(in_b[2*d:])
            params[f'{pfx}_o_weight'] = quant_weight(layer.self_attn.out_proj.weight)
            params[f'{pfx}_o_bias'] = quant_bias(layer.self_attn.out_proj.bias)

            # FFN
            params[f'{pfx}_ff1_weight'] = quant_weight(layer.linear1.weight)
            params[f'{pfx}_ff1_bias'] = quant_bias(layer.linear1.bias)
            params[f'{pfx}_ff2_weight'] = quant_weight(layer.linear2.weight)
            params[f'{pfx}_ff2_bias'] = quant_bias(layer.linear2.bias)

            # Layer norms: quantize to i16 (scale 256) for integer kernel
            params[f'{pfx}_ln1_w'] = quant_ln(layer.norm1.weight)
            params[f'{pfx}_ln1_b'] = quant_ln(layer.norm1.bias)
            params[f'{pfx}_ln2_w'] = quant_ln(layer.norm2.weight)
            params[f'{pfx}_ln2_b'] = quant_ln(layer.norm2.bias)

        # Final layer norm + output head
        params['lnf_w'] = quant_ln(self.ln_final.weight)
        params['lnf_b'] = quant_ln(self.ln_final.bias)
        params['head_weight'] = quant_weight(self.head.weight)

        return params


# ─── Tokenization ───────────────────────────────────────────────────

def encode(text: str) -> List[int]:
    """Convert ASCII string to byte tokens."""
    return [b for b in text.encode('ascii', errors='replace')]


def decode(tokens: List[int]) -> str:
    """Convert byte tokens back to string."""
    return bytes(b for b in tokens if b < 256).decode('ascii', errors='replace')


def make_sequence(query: str, response: str, max_len: int = DEFAULT_MAX_LEN,
                  truncate: bool = False) -> Tuple[List[int], List[int]]:
    """Create input/target pair for autoregressive training.

    Layout: [Q bytes] [SEP] [R bytes] [EOS]

    SEP (byte 1) separates query from response so the model learns a clean
    response zone.

    truncate=False (default): pairs that exceed max_len return [] — caller
      drops them so EOS always lands at a natural sentence end. Correct when
      enough short pairs exist (Shade/Specter where ctx is large).

    truncate=True: over-long pairs truncate R to fit rather than being dropped.
      EOS lands mid-sentence but SEP still teaches clean zone boundaries,
      preventing the doubled-response issue. Use for Wisp (ctx=64) where
      strict filtering would discard 97% of diverse training data.
    """
    q_bytes = encode(query)
    r_bytes = encode(response)

    full = q_bytes + [SEP_TOKEN] + r_bytes
    if len(full) + 1 > max_len:
        if not truncate:
            return [], []          # signal caller to drop
        full = full[:max_len - 1]  # truncate R, EOS will follow

    # Input: Q + SEP + R + EOS
    inp = full + [EOS_TOKEN]

    # Target: predict next token (shifted left), EOS marks end, PAD fills remainder
    tgt = full[1:] + [EOS_TOKEN, PAD_TOKEN]
    tgt = tgt[:len(inp)]

    return inp, tgt


# ─── Training ───────────────────────────────────────────────────────

def split_pairs(pairs, val_frac: float, seed: int = 0):
    """Deterministically split pairs into (train, val). Always leaves >=1 train."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    pairs = list(pairs)
    if val_frac == 0.0 or len(pairs) < 2:
        return pairs, []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_val = min(max(1, round(len(pairs) * val_frac)), len(pairs) - 1)
    val_set = set(order[:n_val].tolist())
    train = [p for i, p in enumerate(pairs) if i not in val_set]
    val = [p for i, p in enumerate(pairs) if i in val_set]
    return train, val


def _build_sequences(pairs, max_len: int, preserve_case: bool = False, truncate: bool = False):
    """Tokenize pairs into (inputs, targets), dropping empty / too-short ones."""
    inputs, targets = [], []
    for q, r in pairs:
        if not q or not r:
            continue
        q_norm = q.strip() if preserve_case else q.upper().strip()
        r_norm = r.strip() if preserve_case else r.upper().strip()
        inp, tgt = make_sequence(q_norm, r_norm, max_len, truncate=truncate)
        # make_sequence returns [] when truncate=False and pair is over-length.
        if len(inp) >= 4:
            inputs.append(inp)
            targets.append(tgt)
    return inputs, targets


def parse_multiturn_line(line: str) -> Optional[List[Tuple[str, str]]]:
    """Parse a multi-turn data line into a list of (query, response) pairs.

    Format: Q1|R1|Q2|R2|...  (even number of pipe-separated fields).
    Single-turn Q|R is also valid (returns a one-element list).
    Odd trailing fields are dropped. Returns None for unparseable lines.
    """
    if not line or '|' not in line:
        return None
    parts = line.strip().split('|')
    if len(parts) % 2 != 0:
        parts = parts[:-1]
    if len(parts) < 2:
        return None
    return [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]


def make_sequence_multiturn(
    turns: List[Tuple[str, str]],
    max_len: int = DEFAULT_MAX_LEN,
) -> Tuple[List[int], List[int]]:
    """Build an autoregressive sequence from a list of (query, response) pairs.

    Layout: [Q1][SEP][R1][SEP][Q2][SEP][R2][EOS]

    Returns ([], []) when the sequence exceeds max_len.
    For a single turn this produces an identical result to make_sequence().
    """
    tokens: List[int] = []
    for q, r in turns:
        tokens += encode(q) + [SEP_TOKEN] + encode(r) + [SEP_TOKEN]
    # Replace trailing SEP with EOS
    if not tokens:
        return [], []
    tokens[-1] = EOS_TOKEN

    if len(tokens) > max_len:
        return [], []

    inp = tokens
    tgt = tokens[1:] + [PAD_TOKEN]   # shift-left; last slot predicts PAD (ignored in loss)
    return inp, tgt


def make_batches(items, batch_size: int):
    """Split a sequence into consecutive batches of at most batch_size."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def train_transformer(
    model: TinyTransformer,
    pairs: List[Tuple[str, str]],
    epochs: int = 500,
    lr: float = 0.003,
    device: str = 'cuda',
    checkpoint_file: str = 'transformer_model.pt',
    batch_size: int = 16,
    amp: bool = False,
    qat_every: int = 1,
    qat_weight: float = 0.10,
    qat_bits: int = 4,
    val_frac: float = 0.0,
    patience: int = 0,
    status_file: Optional[str] = None,
    preserve_case: bool = False,
    truncate: bool = False,
):
    """Returns a dict: {model, epochs_run, best_val_loss, stopped_early}."""
    # Lazy-import supervision emitter — no hard dep when not used.
    _write_status = None
    if status_file:
        try:
            from training_status import write_status as _ws
        except ImportError:
            import importlib.util
            _spec = importlib.util.spec_from_file_location(
                "training_status",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_status.py"),
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _ws = _mod.write_status
        _write_status = _ws

    print(f"Device: {device}")
    print(f"Training on {len(pairs)} pairs, {epochs} epochs")

    model = model.to(device)
    model.train()

    train_pairs, val_pairs = split_pairs(pairs, val_frac)
    all_inputs, all_targets = _build_sequences(train_pairs, model.max_len, preserve_case, truncate)
    val_inputs, val_targets = _build_sequences(val_pairs, model.max_len, preserve_case, truncate)

    print(f"Sequences — train: {len(all_inputs)}, val: {len(val_inputs)}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    use_amp = amp and str(device).startswith('cuda')
    if use_amp:
        print("Mixed precision: bf16 autocast enabled")

    def autocast_ctx():
        return (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                if use_amp else nullcontext())

    train_start = time.time()
    best_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    step = 0
    stopped_early = False

    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0

        idxs = np.random.permutation(len(all_inputs))

        for batch_idxs in make_batches(idxs, batch_size):
            batch_inputs = [all_inputs[i] for i in batch_idxs]
            batch_targets = [all_targets[i] for i in batch_idxs]

            # Pad to max length in batch
            max_blen = max(len(inp) for inp in batch_inputs)
            padded_x = []
            padded_y = []
            for inp, tgt in zip(batch_inputs, batch_targets):
                pad_n = max_blen - len(inp)
                padded_x.append(inp + [PAD_TOKEN] * pad_n)
                padded_y.append(tgt + [PAD_TOKEN] * pad_n)

            x = torch.tensor(padded_x, dtype=torch.long, device=device)
            y = torch.tensor(padded_y, dtype=torch.long, device=device)

            optimizer.zero_grad()

            with autocast_ctx():
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    y.reshape(-1),
                    ignore_index=PAD_TOKEN,
                )

            # Quantization-aware penalty is expensive (per-tensor quantile), so
            # apply it only every `qat_every` steps. qat_every<=0 disables it.
            if qat_every > 0 and step % qat_every == 0:
                total = loss + model.compute_quantization_loss(qat_bits) * qat_weight
            else:
                total = loss
            total.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred_tokens = logits.argmax(dim=-1)
            mask = y != PAD_TOKEN
            total_correct += (pred_tokens[mask] == y[mask]).sum().item()
            total_tokens += mask.sum().item()
            n_batches += 1
            step += 1

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        acc = total_correct / max(total_tokens, 1)

        # Validation pass (no grad, no AMP needed — we want consistent fp32 loss)
        val_loss = None
        if val_inputs:
            model.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_idxs in make_batches(list(range(len(val_inputs))), batch_size):
                    max_blen = max(len(val_inputs[i]) for i in batch_idxs)
                    px = [val_inputs[i] + [PAD_TOKEN] * (max_blen - len(val_inputs[i]))
                          for i in batch_idxs]
                    py = [val_targets[i] + [PAD_TOKEN] * (max_blen - len(val_targets[i]))
                          for i in batch_idxs]
                    x = torch.tensor(px, dtype=torch.long, device=device)
                    y = torch.tensor(py, dtype=torch.long, device=device)
                    logits = model(x)
                    val_total += F.cross_entropy(
                        logits.reshape(-1, model.vocab_size),
                        y.reshape(-1),
                        ignore_index=PAD_TOKEN,
                    ).item()
                    val_batches += 1
            val_loss = val_total / max(val_batches, 1)
            model.train()

        if (epoch + 1) % 5 == 0:
            msg = f"  Epoch {epoch + 1:4d}/{epochs}: loss={avg_loss:.4f}, acc={acc:.3f}"
            if val_loss is not None:
                msg += f", val_loss={val_loss:.4f}"
            print(msg)

        # Checkpoint on best val loss when a val set exists; else best train acc.
        improved = (val_loss is not None and val_loss < best_val_loss) or \
                   (val_loss is None and acc > best_acc)
        if improved:
            if val_loss is not None:
                best_val_loss = val_loss
            best_acc = acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'architecture': {
                    'type': 'tiny_transformer',
                    'vocab_size': model.vocab_size,
                    'd_model': model.d_model,
                    'n_heads': model.n_heads,
                    'n_layers': model.n_layers,
                    'd_ff': model.d_ff,
                    'max_len': model.max_len,
                    'weight_bits': 4,
                },
                'best_acc': best_acc,
                'best_val_loss': best_val_loss if val_inputs else None,
                'best_epoch': best_epoch,
                'epoch': epoch + 1,
            }, checkpoint_file)
        else:
            no_improve += 1

        if _write_status and status_file:
            elapsed = time.time() - train_start
            ep_done = epoch + 1
            eta = int(elapsed / ep_done * (epochs - ep_done)) if ep_done else 0
            _write_status(
                status_file,
                epoch=ep_done, epochs_total=epochs,
                train_loss=round(avg_loss, 6),
                val_loss=round(val_loss, 6) if val_loss is not None else None,
                best_val_loss=round(best_val_loss, 6) if val_inputs else None,
                best_epoch=best_epoch, stopped_early=False,
                state="running", pid=os.getpid(),
                checkpoint=checkpoint_file, eta_seconds=eta,
            )

        if patience > 0 and no_improve >= patience:
            print(f"\nEarly stop at epoch {epoch + 1} (no val improvement for {patience} epochs)")
            stopped_early = True
            break

    if _write_status and status_file:
        _write_status(
            status_file,
            epoch=epoch + 1, epochs_total=epochs,
            stopped_early=stopped_early,
            state="early_stopped" if stopped_early else "done",
            pid=os.getpid(), checkpoint=checkpoint_file, eta_seconds=0,
        )

    print(f"\nBest acc: {best_acc:.3f} at epoch {best_epoch}")
    return {
        'model': model,
        'epochs_run': epoch + 1,
        'best_val_loss': best_val_loss if val_inputs else None,
        'stopped_early': stopped_early,
    }


# ─── Generation ─────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: TinyTransformer,
    prompt: str,
    max_new: int = 60,
    temperature: float = 0.8,
    device: str = 'cpu',
    preserve_case: bool = False,
) -> str:
    model.eval()
    model = model.to(device)

    # Cap the prompt at the context window (mirrors the TS orchestrator). The old
    # `[:max_len - max_new]` silently dropped prompt bytes (e.g. "HELLO" → "HELL").
    prompt_norm = prompt if preserve_case else prompt.upper()
    # Inject SEP after the prompt — this puts the model in "response zone"
    # so it generates R directly rather than potentially emitting SEP first.
    tokens = encode(prompt_norm)[:model.max_len - 2] + [SEP_TOKEN]
    prompt_len = len(tokens)

    for _ in range(max_new):
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(x)
        next_logits = logits[0, -1] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        if next_token == EOS_TOKEN or next_token == PAD_TOKEN:
            break
        # Skip any SEP token the model emits (shouldn't happen with injected SEP,
        # but guard against it so it never appears in output.)
        if next_token != SEP_TOKEN:
            tokens.append(next_token)
        if len(tokens) >= model.max_len:
            break

    # Return only the generated portion (after the prompt tokens we kept).
    # decode() already filters SEP (byte 1) since it only outputs b < 256 that
    # are printable; SEP_TOKEN=1 is a control char that bytes().decode replaces.
    return decode(tokens[prompt_len:])


# ─── CLI ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, required=True)
    parser.add_argument('--epochs', '-e', type=int, default=500)
    parser.add_argument('--device', '-d', type=str, default='auto')
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=1024)
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default='transformer_model.pt')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='recommended 64-256 on the 5080 for the small tiers')
    parser.add_argument('--amp', action='store_true',
                        help='bf16 autocast (CUDA only; no-op on CPU)')
    parser.add_argument('--qat-every', type=int, default=1,
                        help='apply the quantization penalty every N steps (0=off)')
    parser.add_argument('--qat-weight', type=float, default=0.10)
    parser.add_argument('--qat-bits', type=int, default=4, choices=[4, 8],
                        help='target bit-width for QAT loss (4 or 8)')
    parser.add_argument('--val-frac', type=float, default=0.0,
                        help='fraction of pairs held out for validation (e.g. 0.05)')
    parser.add_argument('--patience', type=int, default=0,
                        help='early-stop after N epochs of no val improvement (0=off)')
    parser.add_argument('--status-file', type=str, default=None,
                        help='path to write status.json each epoch (supervision harness)')
    parser.add_argument('--preserve-case', action='store_true',
                        help='disable uppercase normalization (required for Wraith/technical models)')
    parser.add_argument('--truncate', action='store_true',
                        help='truncate over-length pairs instead of dropping them (use for small ctx like Wisp)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load training data
    pairs = []
    with open(args.file) as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            q, r = line.split('|', 1)
            pairs.append((q.strip(), r.strip()))

    print(f"Loaded {len(pairs)} training pairs")

    # Create or resume model
    if args.resume and __import__('os').path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        arch = ckpt['architecture']
        model = TinyTransformer(
            vocab_size=arch['vocab_size'],
            d_model=arch['d_model'],
            n_heads=arch['n_heads'],
            n_layers=arch['n_layers'],
            d_ff=arch['d_ff'],
            max_len=arch['max_len'],
        )
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0)
        remaining = max(1, args.epochs - start_epoch)
        print(f"Resuming from epoch {start_epoch}, {remaining} epochs remaining")
    else:
        model = TinyTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_len=args.max_len,
        )
        remaining = args.epochs

    result = train_transformer(
        model, pairs, epochs=remaining, lr=args.lr,
        device=device, checkpoint_file=args.checkpoint,
        batch_size=args.batch_size, amp=args.amp,
        qat_every=args.qat_every, qat_weight=args.qat_weight, qat_bits=args.qat_bits,
        val_frac=args.val_frac, patience=args.patience,
        status_file=args.status_file,
        preserve_case=args.preserve_case,
        truncate=args.truncate,
    )
    model = result['model']

    # Test generation
    print("\nSample generations:")
    for prompt in ['HELLO', 'HOW ARE YOU', 'TELL ME A JOKE', 'WHO ARE YOU', 'THANKS']:
        out = generate(model, prompt, 50, temperature=0.8, device=device)
        print(f"  {prompt:20s} → {out}")
