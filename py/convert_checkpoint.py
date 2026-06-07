#!/usr/bin/env python3
"""Convert the pre-trained z80ai checkpoint to a Tier 2 browser bundle."""

import sys
sys.path.insert(0, ".")

import torch
from tier2_serialization import AutoregressiveModel, serialize_model


def convert_checkpoint(ckpt_path: str, output_prefix: str):
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, weights_only=True)

    arch = ckpt["architecture"]
    charset = ckpt["charset"]
    input_size = arch["input_size"]
    hidden_sizes = arch["hidden_sizes"]
    num_chars = arch["num_classes"]  # 40

    print(f"  Architecture: {input_size} → {'→'.join(map(str, hidden_sizes))} → {num_chars}")
    print(f"  Charset ({num_chars} chars): {repr(charset[:30])}...")
    print(f"  Trained: {ckpt.get('total_epochs', '?')} epochs, "
          f"best int acc: {ckpt.get('best_int_acc', 0):.1%}")

    # Build model and load state
    model = AutoregressiveModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_chars=num_chars,
    )

    # The original uses OverflowAwareLinear which has extra keys like
    # max_accum_seen. Filter those out so we only load weight/bias.
    state_dict = ckpt["model_state"]
    clean_state = {}
    for k, v in state_dict.items():
        if "max_accum_seen" not in k:
            clean_state[k] = v

    model.load_state_dict(clean_state, strict=False)
    model.eval()

    # Test a quick forward pass to make sure it works
    with torch.no_grad():
        dummy = torch.randn(1, 256)
        out = model(dummy)
        top = out.argmax(dim=1).item()
        top_char = charset[top] if top < len(charset) else "?"
        print(f"  Quick forward pass: top char = {repr(top_char)} (idx {top})")

    # Get weight_bits from checkpoint, default to 2
    weight_bits = arch.get("weight_bits", 2)
    print(f"  Weight bits: {weight_bits}")

    # Serialize
    serialize_model(model, charset, output_prefix, weight_bits=weight_bits)

    print(f"\nDone. Files: {output_prefix}.bin, {output_prefix}.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c",
                        default="/home/alex/dev/ai/z80/z80ai/examples/tinychat/command_model_autoreg.pt")
    parser.add_argument("--output", "-o", default="stage3/dist/trained_model")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output)
