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


# ─── Modern Architecture Components ────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean-centering, no bias).

    Faster than LayerNorm and empirically as good. Used in LLaMA/Gemma.
    """
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: SiLU(W1·x) ⊙ W2·x → W3.

    Better gradient flow than ReLU. Used in LLaMA/Gemma/PaLM.
    No biases — consistent with modern LLM practice.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate branch
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # value branch
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def precompute_freqs_cis(d_head: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute complex rotary frequencies for RoPE.

    Returns complex tensor of shape (max_len, d_head//2).
    """
    freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K tensors.

    xq, xk: (batch, seq, n_heads, d_head)
    freqs_cis: (seq, d_head//2) complex
    """
    def rotate(x: torch.Tensor) -> torch.Tensor:
        B, T, H, D = x.shape
        xc = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))
        fc = freqs_cis[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D//2)
        xr = torch.view_as_real(xc * fc).reshape(B, T, H, D)
        return xr.type_as(x)
    return rotate(xq), rotate(xk)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE."""
    def __init__(self, d_model: int, n_heads: int, use_rope: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.use_rope = use_rope
        self.dropout  = dropout

        self.in_proj  = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.in_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if self.use_rope and freqs_cis is not None:
            # reshape to (B, T, H, D) for apply_rotary_emb
            q_r = q.transpose(1, 2)
            k_r = k.transpose(1, 2)
            q_r, k_r = apply_rotary_emb(q_r, k_r, freqs_cis[:T])
            q = q_r.transpose(1, 2)
            k = k_r.transpose(1, 2)

        # Use PyTorch's flash-attention-compatible SDPA
        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class ModernTransformerBlock(nn.Module):
    """Transformer block with RMSNorm, SwiGLU, and optional RoPE."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 use_rope: bool = True, use_swiglu: bool = True,
                 use_rmsnorm: bool = True, dropout: float = 0.1):
        super().__init__()
        NormCls = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = NormCls(d_model)
        self.norm2 = NormCls(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, use_rope=use_rope,
                                         dropout=dropout)
        if use_swiglu:
            self.ff = SwiGLUFFN(d_model, d_ff)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(),
                nn.Linear(d_ff, d_model),
            )

    def forward(self, x: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x


class TinyTransformerModern(nn.Module):
    """Modern transformer with RMSNorm, SwiGLU, RoPE, and optional weight tying.

    Drop-in replacement for TinyTransformer with improved components:
    - RMSNorm: faster, no mean centering, no bias
    - SwiGLU: gated FFN, better gradient flow
    - RoPE: rotary position embeddings, better length generalisation
    - Weight tying: shares token embedding and output head weights
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = DEFAULT_MAX_LEN,
        dropout: float = 0.1,
        use_rope:    bool = True,
        use_swiglu:  bool = True,
        use_rmsnorm: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.d_ff       = d_ff
        self.max_len    = max_len
        self.use_rope   = use_rope

        self.token_embed = nn.Embedding(vocab_size, d_model)

        if use_rope:
            self.register_buffer(
                'freqs_cis',
                precompute_freqs_cis(d_model // n_heads, max_len),
                persistent=False,
            )
        else:
            self.pos_embed = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            ModernTransformerBlock(d_model, n_heads, d_ff,
                                   use_rope=use_rope, use_swiglu=use_swiglu,
                                   use_rmsnorm=use_rmsnorm, dropout=dropout)
            for _ in range(n_layers)
        ])

        NormCls = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln_final = NormCls(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'token_embed' not in name:
                nn.init.xavier_uniform_(p, gain=0.5)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def compute_quantization_loss(self, weight_bits: int = 4) -> torch.Tensor:
        max_int = float(2 ** (weight_bits - 1) - 1)
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            if p.dim() <= 1:
                continue
            absmax = p.detach().abs().max().clamp(min=1e-8)
            scale = absmax / max_int
            quantized = (p / scale).round().clamp(-max_int, max_int) * scale
            loss = loss + (p - quantized).pow(2).mean()
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.token_embed(x) * math.sqrt(self.d_model)

        if self.use_rope:
            freqs = self.freqs_cis[:T]
            for block in self.blocks:
                h = block(h, freqs)
        else:
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            h = h + self.pos_embed(pos)
            for block in self.blocks:
                h = block(h)

        h = self.ln_final(h)
        return self.head(h)


# ─── Ternary architecture ───────────────────────────────────────────

class TernaryLinear(nn.Module):
    """Linear layer that pushes weights toward {-scale, 0, +scale} via STE.

    Forward pass: weights are ternarized using absmean thresholding.
    Backward pass: gradients flow through the continuous fp32 weights
    (straight-through estimator) so the optimizer can move them.

    Scale = absmean(weight). Threshold = 0.5 * scale.
    Values with |w| < threshold collapse to 0; others become ±scale.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def ternarize(self, w: torch.Tensor) -> torch.Tensor:
        scale     = w.abs().mean().clamp(min=1e-8)
        threshold = 0.5 * scale
        return scale * torch.where(w.abs() < threshold,
                                   torch.zeros_like(w),
                                   torch.sign(w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # STE: use ternary values in forward, let gradient pass through fp32 weight
        w_t = self.weight + (self.ternarize(self.weight) - self.weight).detach()
        return F.linear(x, w_t, self.bias)


class _TernaryAttention(nn.Module):
    """Multi-head causal self-attention with ternary Q/K/V/O projections."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.q_proj  = TernaryLinear(d_model, d_model)
        self.k_proj  = TernaryLinear(d_model, d_model)
        self.v_proj  = TernaryLinear(d_model, d_model)
        self.o_proj  = TernaryLinear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scale  = math.sqrt(self.d_head)
        scores = (q @ k.transpose(-2, -1)) / scale
        mask   = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        out    = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.o_proj(out)


class _TernaryFFN(nn.Module):
    """Position-wise FFN: TernaryLinear → ReLU → TernaryLinear."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = TernaryLinear(d_model, d_ff)
        self.w2 = TernaryLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


class _TernaryBlock(nn.Module):
    """Pre-norm transformer block with ternary attention and FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = _TernaryAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = _TernaryFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyTransformerTernary(nn.Module):
    """Wisp-scale causal transformer with ternary linear weights.

    Same structure as TinyTransformer (classic) but all attention and FFN
    projections use TernaryLinear with straight-through estimator training.
    Embeddings, layer norms, and biases remain fp32.

    Intended for fast experimentation on a secondary GPU before committing
    to full Revenant (300M) training.
    """

    arch = 'ternary'

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.d_ff       = d_ff
        self.max_len    = max_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.blocks      = nn.ModuleList([
            _TernaryBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def compute_quantization_loss(self, weight_bits: int = 4) -> torch.Tensor:
        # STE handles ternarization natively — no extra QAT penalty needed
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h   = self.token_embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_final(h)
        return self.head(h)


# ─── Sequence builders ──────────────────────────────────────────────

def make_sequence(query: str, response: str, max_len: int = DEFAULT_MAX_LEN,
                  truncate: bool = False, mask_query: bool = False,
                  tok=None) -> Tuple[List[int], List[int]]:
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
    if tok is not None:
        q_bytes = tok.encode(query)
        r_bytes = tok.encode(response)
        _PAD, _EOS, _SEP = tok.PAD, tok.EOS, tok.SEP
    else:
        q_bytes = encode(query)
        r_bytes = encode(response)
        _PAD, _EOS, _SEP = PAD_TOKEN, EOS_TOKEN, SEP_TOKEN

    full = q_bytes + [_SEP] + r_bytes
    if len(full) + 1 > max_len:
        if not truncate:
            return [], []
        full = full[:max_len - 1]

    inp = full + [_EOS]
    tgt = full[1:] + [_EOS, _PAD]
    tgt = tgt[:len(inp)]

    if mask_query:
        for i in range(min(len(q_bytes), len(tgt))):
            tgt[i] = _PAD

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


def _build_sequences(pairs, max_len: int, preserve_case: bool = False,
                     truncate: bool = False, mask_query: bool = False, tok=None):
    """Tokenize pairs into (inputs, targets), dropping empty / too-short ones.

    Accepts either single-turn (q, r) tuples or multi-turn [(q1,r1),(q2,r2),...]
    lists. Multi-turn items use make_sequence_multiturn(); single-turn items use
    make_sequence() for backward compatibility with existing callers.
    """
    inputs, targets = [], []
    for item in pairs:
        turns = [item] if isinstance(item, tuple) else list(item)
        if not turns or not all(q and r for q, r in turns):
            continue
        if not preserve_case:
            turns = [(q.upper().strip(), r.upper().strip()) for q, r in turns]
        else:
            turns = [(q.strip(), r.strip()) for q, r in turns]
        if len(turns) == 1:
            inp, tgt = make_sequence(turns[0][0], turns[0][1], max_len,
                                     truncate=truncate, mask_query=mask_query, tok=tok)
        else:
            inp, tgt = make_sequence_multiturn(turns, max_len, mask_query=mask_query, tok=tok)
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
    mask_query: bool = False,
    tok=None,
) -> Tuple[List[int], List[int]]:
    """Build an autoregressive sequence from a list of (query, response) pairs.

    Layout: [Q1][SEP][R1][SEP][Q2][SEP][R2][EOS]

    Returns ([], []) when the sequence exceeds max_len.
    For a single turn this produces an identical result to make_sequence().

    mask_query: if True, query byte positions in the target are set to PAD_TOKEN
      so the loss is only computed on response tokens.
    """
    _PAD = tok.PAD if tok else PAD_TOKEN
    _EOS = tok.EOS if tok else EOS_TOKEN
    _SEP = tok.SEP if tok else SEP_TOKEN
    _enc = tok.encode if tok else encode

    tokens: List[int] = []
    is_query_pos: List[bool] = []
    for q, r in turns:
        q_bytes = _enc(q)
        r_bytes = _enc(r)
        tokens += q_bytes + [_SEP] + r_bytes + [_SEP]
        is_query_pos += [True] * len(q_bytes) + [False] * (1 + len(r_bytes) + 1)

    if not tokens:
        return [], []
    tokens[-1] = _EOS

    if len(tokens) > max_len:
        return [], []

    inp = tokens
    tgt = tokens[1:] + [_PAD]

    if mask_query:
        for i in range(len(tgt)):
            if i < len(is_query_pos) and is_query_pos[i]:
                tgt[i] = _PAD

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
    mask_query: bool = False,
    tok=None,
    resume_best_val_loss=None,
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

    from datetime import datetime
    print(f"Device: {device}")
    print(f"Training on {len(pairs)} pairs, {epochs} epochs")
    print(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    model = model.to(device)
    model.train()

    pad_id = tok.PAD if tok else PAD_TOKEN

    train_pairs, val_pairs = split_pairs(pairs, val_frac)
    all_inputs, all_targets = _build_sequences(train_pairs, model.max_len, preserve_case, truncate, mask_query, tok)
    val_inputs, val_targets = _build_sequences(val_pairs, model.max_len, preserve_case, truncate, mask_query, tok)

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
    best_val_loss = resume_best_val_loss if resume_best_val_loss is not None else float('inf')
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
                padded_x.append(inp + [pad_id] * pad_n)
                padded_y.append(tgt + [pad_id] * pad_n)

            x = torch.tensor(padded_x, dtype=torch.long, device=device)
            y = torch.tensor(padded_y, dtype=torch.long, device=device)

            optimizer.zero_grad()

            with autocast_ctx():
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    y.reshape(-1),
                    ignore_index=pad_id,
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
            mask = y != pad_id
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
                    px = [val_inputs[i] + [pad_id] * (max_blen - len(val_inputs[i]))
                          for i in batch_idxs]
                    py = [val_targets[i] + [pad_id] * (max_blen - len(val_targets[i]))
                          for i in batch_idxs]
                    x = torch.tensor(px, dtype=torch.long, device=device)
                    y = torch.tensor(py, dtype=torch.long, device=device)
                    logits = model(x)
                    val_total += F.cross_entropy(
                        logits.reshape(-1, model.vocab_size),
                        y.reshape(-1),
                        ignore_index=pad_id,
                    ).item()
                    val_batches += 1
            val_loss = val_total / max(val_batches, 1)
            model.train()

        if (epoch + 1) % 5 == 0:
            from datetime import datetime
            ts = datetime.utcnow().strftime('%H:%M:%S')
            msg = f"  [{ts}] Epoch {epoch + 1:4d}/{epochs}: loss={avg_loss:.4f}, acc={acc:.3f}"
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
                    'arch': ('ternary' if isinstance(model, TinyTransformerTernary)
                             else 'modern' if isinstance(model, TinyTransformerModern)
                             else 'classic'),
                    # Save individual modern flags so loaders don't need to infer from state dict
                    **({'use_rope':     model.use_rope,
                        'use_swiglu':   any(hasattr(b.ff, 'w1') for b in model.blocks),
                        'use_rmsnorm':  isinstance(model.blocks[0].norm1, RMSNorm),
                        'tie_weights':  model.head.weight is model.token_embed.weight,
                       } if isinstance(model, TinyTransformerModern) else {}),
                    'vocab_size': model.vocab_size,
                    'd_model': model.d_model,
                    'n_heads': model.n_heads,
                    'n_layers': model.n_layers,
                    'd_ff': model.d_ff,
                    'max_len': model.max_len,
                    'weight_bits': 4,
                    **(({'tokenizer_file': checkpoint_file.replace('.pt', '_tokenizer.json'),
                         'tokenizer_type': 'bpe'}) if tok else {}),
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

    print(f"\nFinished: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Best acc: {best_acc:.3f} at epoch {best_epoch}")
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
    parser.add_argument('--arch', default='classic', choices=['classic', 'modern', 'ternary'],
                        help='classic = original TinyTransformer; modern = RMSNorm+SwiGLU+RoPE+weight-tying')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='path to BPE tokenizer JSON (from train_bpe.py); omit for byte-level')
    parser.add_argument('--mask-query-loss', action='store_true',
                        help='only compute loss on response tokens (masks query positions)')
    # Fine-grained modern arch overrides (only apply when --arch modern)
    parser.add_argument('--no-rope',      action='store_true', help='disable RoPE (use learned pos embed)')
    parser.add_argument('--no-swiglu',    action='store_true', help='disable SwiGLU (use ReLU FFN)')
    parser.add_argument('--no-rmsnorm',   action='store_true', help='disable RMSNorm (use LayerNorm)')
    parser.add_argument('--no-tie-weights', action='store_true', help='disable weight tying')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # BPE tokenizer (optional — byte-level used when absent)
    tok = None
    if args.tokenizer:
        import importlib.util, sys as _sys
        _spec = importlib.util.spec_from_file_location(
            'bpe_tokenizer',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bpe_tokenizer.py'),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        tok = _mod.BPETokenizer(args.tokenizer)
        print(f"BPE tokenizer loaded: {tok.vocab_size} tokens from {args.tokenizer}")

    # Load training data — supports both single-turn (Q|R) and multi-turn (Q1|R1|Q2|R2|...) lines.
    pairs = []
    n_multiturn = 0
    with open(args.file) as f:
        for line in f:
            line = line.strip()
            turns = parse_multiturn_line(line)
            if not turns:
                continue
            if len(turns) == 1:
                pairs.append(turns[0])  # (q, r) tuple — handled by existing path
            else:
                pairs.append(turns)     # list of (q,r) — multi-turn path
                n_multiturn += 1

    print(f"Loaded {len(pairs)} training items ({n_multiturn} multi-turn)")

    # Create or resume model
    if args.resume and __import__('os').path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        arch = ckpt['architecture']
        _kw = dict(vocab_size=arch['vocab_size'], d_model=arch['d_model'],
                   n_heads=arch['n_heads'], n_layers=arch['n_layers'],
                   d_ff=arch['d_ff'], max_len=arch['max_len'])
        if arch.get('arch') == 'ternary':
            model = TinyTransformerTernary(**_kw)
        elif arch.get('arch') == 'modern':
            state = ckpt['model_state']
            model = TinyTransformerModern(**_kw,
                use_rope='pos_embed.weight' not in state,
                use_swiglu=any('ff.w1' in k for k in state),
                use_rmsnorm=not any('norm1.bias' in k for k in state),
                tie_weights='head.weight' not in state)
        else:
            model = TinyTransformer(**_kw)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0)
        remaining = max(1, args.epochs - start_epoch)
        # Preserve best_val_loss from checkpoint so resumed runs never overwrite
        # a better checkpoint just because their first epoch beats float('inf')
        _resume_best_val_loss = ckpt.get('best_val_loss', None)
        print(f"Resuming from epoch {start_epoch}, {remaining} epochs remaining")
    else:
        vocab_sz = tok.vocab_size if tok else VOCAB_SIZE
        if args.arch == 'ternary':
            model = TinyTransformerTernary(
                vocab_size=vocab_sz,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                max_len=args.max_len,
            )
        elif args.arch == 'modern':
            model = TinyTransformerModern(
                vocab_size=vocab_sz,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                max_len=args.max_len,
                use_rope=not args.no_rope,
                use_swiglu=not args.no_swiglu,
                use_rmsnorm=not args.no_rmsnorm,
                tie_weights=not args.no_tie_weights,
            )
        else:
            model = TinyTransformer(
                vocab_size=vocab_sz,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                max_len=args.max_len,
            )
        remaining = args.epochs

    # Copy tokenizer JSON next to checkpoint so serialize.py can find it
    if tok and args.tokenizer:
        import shutil
        tok_dest = args.checkpoint.replace('.pt', '_tokenizer.json')
        if os.path.abspath(args.tokenizer) != os.path.abspath(tok_dest):
            shutil.copy(args.tokenizer, tok_dest)
            print(f"Tokenizer copied to {tok_dest}")

    result = train_transformer(
        model, pairs, epochs=remaining, lr=args.lr,
        device=device, checkpoint_file=args.checkpoint,
        batch_size=args.batch_size, amp=args.amp,
        qat_every=args.qat_every, qat_weight=args.qat_weight, qat_bits=args.qat_bits,
        val_frac=args.val_frac, patience=args.patience,
        status_file=args.status_file,
        preserve_case=args.preserve_case,
        truncate=args.truncate,
        mask_query=args.mask_query_loss,
        tok=tok,
        resume_best_val_loss=locals().get('_resume_best_val_loss'),
    )
    model = result['model']

    # Test generation (BPE models use tok.decode)
    print("\nSample generations:")
    for prompt in ['HELLO', 'HOW ARE YOU', 'TELL ME A JOKE', 'WHO ARE YOU', 'THANKS']:
        out = generate(model, prompt, 50, temperature=0.8, device=device)
        print(f"  {prompt:20s} → {out}")
