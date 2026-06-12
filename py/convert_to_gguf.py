#!/usr/bin/env python3
"""Convert a ternary_modern checkpoint to GGUF (Path A: dequantize to F16, llama arch).

Why this maps cleanly onto llama.cpp's `llama` architecture:
  - RMSNorm + RoPE + SwiGLU + no biases  == Llama block structure
  - Our RoPE uses the INTERLEAVED (adjacent-pair / complex) convention, which is
    exactly what llama.cpp's `llama` arch expects (GGML rope type NORM). HF Llama
    uses rotate-half (NEOX) and the HF->GGUF converter PERMUTES q/k to reach the
    interleaved layout — we are already interleaved, so we emit q/k unpermuted.
  - Embedding is scaled by sqrt(d_model) in our forward (a Gemma-ism). llama arch
    does NOT scale embeddings, so we FOLD sqrt(d_model) into token_embd.weight.
    The output head is tied to the *unscaled* embedding, so we emit output.weight
    separately (untied) with the raw values.

Ternary weights are dequantized exactly as training does (absmean threshold) and
stored as F16. No size win vs the browser build — this is a correctness spike to
prove the pipeline; ternary-preserving TQ1_0/TQ2_0 is a later step.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import gguf


def ternarize(w: torch.Tensor) -> torch.Tensor:
    """Dequantize a TernaryLinear weight to its dense {-s,0,+s} matrix (matches training)."""
    scale = w.abs().mean().clamp(min=1e-8)
    threshold = 0.5 * scale
    return scale * torch.where(w.abs() < threshold, torch.zeros_like(w), torch.sign(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    ap.add_argument("--tokenizer", required=True, help="path to bpe tokenizer json")
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--name", default=None, help="model name in metadata")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["architecture"]
    assert cfg["arch"] == "ternary_modern", f"expected ternary_modern, got {cfg['arch']}"
    sd = ck["model_state"]

    d_model = cfg["d_model"]
    n_heads = cfg["n_heads"]
    n_layers = cfg["n_layers"]
    d_ff = cfg["d_ff"]
    max_len = cfg["max_len"]
    vocab_size = cfg["vocab_size"]
    d_head = d_model // n_heads
    name = args.name or os.path.splitext(os.path.basename(args.checkpoint))[0]

    print(f"[convert] {name}: d={d_model} heads={n_heads} layers={n_layers} "
          f"d_ff={d_ff} d_head={d_head} vocab={vocab_size} ctx={max_len}")

    w = gguf.GGUFWriter(args.out, "llama")

    # ── hyperparameters ──
    w.add_name(name)
    w.add_context_length(max_len)
    w.add_embedding_length(d_model)
    w.add_block_count(n_layers)
    w.add_feed_forward_length(d_ff)
    w.add_head_count(n_heads)
    w.add_head_count_kv(n_heads)               # no GQA
    w.add_rope_dimension_count(d_head)         # full-dim rotary
    w.add_layer_norm_rms_eps(1e-5)             # our RMSNorm eps
    w.add_rope_freq_base(10000.0)              # our RoPE base
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    F16 = gguf.GGMLQuantizationType.F16
    F32 = gguf.GGMLQuantizationType.F32

    def add_f16(gguf_name, tensor):
        w.add_tensor(gguf_name, tensor.to(torch.float16).numpy(), raw_dtype=F16)

    def add_f32(gguf_name, tensor):
        # norm weights must stay F32: llama.cpp's RMSNorm multiply rejects f32×f16
        w.add_tensor(gguf_name, tensor.float().numpy(), raw_dtype=F32)

    # ── embeddings / head ──
    tok_embed = sd["token_embed.weight"].float()            # (vocab, d_model), raw (unscaled)
    # token_embd: fold the sqrt(d_model) forward-time scale into the weights
    add_f16("token_embd.weight", tok_embed * math.sqrt(d_model))
    # output head is tied to the UNSCALED embedding -> emit unscaled, untied
    head = sd.get("head.weight", tok_embed).float()
    add_f16("output.weight", head)
    add_f32("output_norm.weight", sd["ln_final.weight"].float())

    # ── per-block tensors ──
    for i in range(n_layers):
        p = f"blocks.{i}."
        b = f"blk.{i}."
        add_f32(b + "attn_norm.weight", sd[p + "norm1.weight"].float())
        add_f16(b + "attn_q.weight", ternarize(sd[p + "attn.q_proj.weight"].float()))
        add_f16(b + "attn_k.weight", ternarize(sd[p + "attn.k_proj.weight"].float()))
        add_f16(b + "attn_v.weight", ternarize(sd[p + "attn.v_proj.weight"].float()))
        add_f16(b + "attn_output.weight", ternarize(sd[p + "attn.o_proj.weight"].float()))
        add_f32(b + "ffn_norm.weight", sd[p + "norm2.weight"].float())
        # SwiGLU: forward is w3( silu(w1(x)) * w2(x) ) -> gate=w1, up=w2, down=w3
        add_f16(b + "ffn_gate.weight", ternarize(sd[p + "ff.w1.weight"].float()))
        add_f16(b + "ffn_up.weight", ternarize(sd[p + "ff.w2.weight"].float()))
        add_f16(b + "ffn_down.weight", ternarize(sd[p + "ff.w3.weight"].float()))

    # ── tokenizer ──
    tk = json.load(open(args.tokenizer))
    if "model" in tk and "added_tokens" in tk:
        # HF byte-level BPE (llama.cpp-compatible) — embed faithfully as gpt-2
        vocab = tk["model"]["vocab"]                       # token_str -> id
        id_to_token = {i: t for t, i in vocab.items()}
        n_tok = len(id_to_token)
        special_ids = {a["id"] for a in tk["added_tokens"] if a.get("special")}
        tokens, toktypes = [], []
        for i in range(n_tok):
            tokens.append(id_to_token[i])
            toktypes.append(gguf.TokenType.CONTROL if i in special_ids
                            else gguf.TokenType.NORMAL)
        merges = [m if isinstance(m, str) else f"{m[0]} {m[1]}"
                  for m in tk["model"]["merges"]]
        pre = "gpt-2"
        eos_id, pad_id = 1, 0
    else:
        # legacy char-level BPE (best-effort; will NOT match llama.cpp tokenization)
        id_to_token = tk["id_to_token"]
        n_tok = len(id_to_token)
        special = tk.get("special", {})
        special_ids = set(special.values())
        tokens, toktypes = [], []
        for i in range(n_tok):
            tokens.append(id_to_token[str(i)])
            toktypes.append(gguf.TokenType.CONTROL if i in special_ids
                            else gguf.TokenType.NORMAL)
        merges = [f"{a} {b}" for a, b in tk["merges"]]
        pre = "default"
        eos_id, pad_id = special.get("eos", 1), special.get("pad", 0)

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(pre)
    w.add_token_list(tokens)
    w.add_token_types(toktypes)
    w.add_token_merges(merges)
    w.add_eos_token_id(eos_id)
    w.add_pad_token_id(pad_id)
    w.add_bos_token_id(eos_id)   # no BOS in our scheme; alias to eos
    w.add_add_bos_token(False)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[convert] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
