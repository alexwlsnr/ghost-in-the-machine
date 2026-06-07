#!/usr/bin/env python3
"""
Byte-level Tiny Transformer training for Tier 2.5 "Ghost Transformer"

Tokenization: 256 byte values + PAD=256
Training: autoregressive on concatenated query+response pairs
Inference: prompt → generate until PAD token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple

# ─── Constants ─────────────────────────────────────────────────────

VOCAB_SIZE = 258  # bytes 0-255 + PAD_TOKEN (256) + EOS_TOKEN (257)
PAD_TOKEN = 256
EOS_TOKEN = 257
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

    def compute_quantization_loss(self) -> torch.Tensor:
        """Encourage weights to stay near quantization levels."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            if p.dim() <= 1:
                continue
            p_flat = p.flatten()
            scale = torch.quantile(p_flat.abs(), 0.95).clamp(min=1e-6)
            p_scaled = p_flat / scale
            loss = loss + F.mse_loss(p_scaled, torch.round(p_scaled))
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


def make_sequence(query: str, response: str, max_len: int = DEFAULT_MAX_LEN) -> Tuple[List[int], List[int]]:
    """Create input/target pair for autoregressive training.

    input:  [Q  bytes...] [R bytes...] [PAD...]
    target: [Q bytes...] [R bytes...] [PAD...]  (shifted left by 1 in training)
    """
    q_bytes = encode(query)
    r_bytes = encode(response)

    full = q_bytes + r_bytes
    if len(full) >= max_len:
        full = full[:max_len - 1]

    # Input: full sequence + EOS (model sees this)
    inp = full + [EOS_TOKEN]

    # Target: predict next token (shifted left), EOS marks end, PAD fills remainder
    tgt = full[1:] + [EOS_TOKEN, PAD_TOKEN]
    tgt = tgt[:len(inp)]

    return inp, tgt


# ─── Training ───────────────────────────────────────────────────────

def train_transformer(
    model: TinyTransformer,
    pairs: List[Tuple[str, str]],
    epochs: int = 500,
    lr: float = 0.003,
    device: str = 'cuda',
    checkpoint_file: str = 'transformer_model.pt',
):
    print(f"Device: {device}")
    print(f"Training on {len(pairs)} pairs, {epochs} epochs")

    model = model.to(device)
    model.train()

    # Prepare all sequences
    all_inputs = []
    all_targets = []
    for q, r in pairs:
        if not q or not r:
            continue
        inp, tgt = make_sequence(q.upper().strip(), r.upper().strip(), model.max_len)
        if len(inp) >= 4:  # need at least a few tokens
            all_inputs.append(inp)
            all_targets.append(tgt)

    print(f"Generated {len(all_inputs)} training sequences")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0

        idxs = np.random.permutation(len(all_inputs))

        for batch_start in range(0, len(idxs), 16):
            batch_idxs = idxs[batch_start:batch_start + 16]
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

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                y.reshape(-1),
                ignore_index=PAD_TOKEN,
            )

            # Add quantization loss
            qloss = model.compute_quantization_loss() * 0.10
            (loss + qloss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred_tokens = logits.argmax(dim=-1)
            mask = y != PAD_TOKEN
            total_correct += (pred_tokens[mask] == y[mask]).sum().item()
            total_tokens += mask.sum().item()
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        acc = total_correct / max(total_tokens, 1)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:4d}/{epochs}: loss={avg_loss:.4f}, acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
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
                'best_epoch': best_epoch,
                'epoch': epoch + 1,
            }, checkpoint_file)

    print(f"\nBest acc: {best_acc:.3f} at epoch {best_epoch}")
    return model


# ─── Generation ─────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: TinyTransformer,
    prompt: str,
    max_new: int = 60,
    temperature: float = 0.8,
    device: str = 'cpu',
) -> str:
    model.eval()
    model = model.to(device)

    # Cap the prompt at the context window (mirrors the TS orchestrator). The old
    # `[:max_len - max_new]` silently dropped prompt bytes (e.g. "HELLO" → "HELL").
    tokens = encode(prompt.upper())[:model.max_len - 1]
    prompt_len = len(tokens)

    for _ in range(max_new):
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(x)
        next_logits = logits[0, -1] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        if next_token == EOS_TOKEN or next_token == PAD_TOKEN:
            break
        tokens.append(next_token)
        if len(tokens) >= model.max_len:
            break

    # Return only the generated portion (after the prompt tokens we kept).
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

    model = train_transformer(
        model, pairs, epochs=remaining, lr=args.lr,
        device=device, checkpoint_file=args.checkpoint,
    )

    # Test generation
    print("\nSample generations:")
    for prompt in ['HELLO', 'HOW ARE YOU', 'TELL ME A JOKE', 'WHO ARE YOU', 'THANKS']:
        out = generate(model, prompt, 50, temperature=0.8, device=device)
        print(f"  {prompt:20s} → {out}")
