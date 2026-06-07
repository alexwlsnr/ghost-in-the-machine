#!/usr/bin/env python3
"""Generate test vectors for the Stage 2 Wasm kernel verification harness.

Produces a JSON file with:
  - Packed weights (base64)
  - Biases (int16 array)
  - Inputs (int16 array)
  - Expected outputs (int16 array) — computed by the Python reference
"""

import argparse
import base64
import json
import numpy as np
import sys

# Add parent dir to path so we can import tier2_serialization
sys.path.insert(0, "..")
from tier2_serialization import pack_2bit_weights, unpack_2bit_weights


def simulate_layer_python(
    weights_i8: np.ndarray,   # (out_dim, in_dim) int8 in {-2,-1,0,+1}
    biases: np.ndarray,       # (out_dim,) int16
    inputs: np.ndarray,       # (in_dim,) int16
) -> np.ndarray:
    """Python reference for the Wasm kernel.

    Mirrors exactly the arithmetic in stage2/src/lib.rs.
    """
    out_dim = len(biases)
    in_dim = len(inputs)
    outputs = np.zeros(out_dim, dtype=np.int16)

    for o in range(out_dim):
        acc: int = 0
        for j in range(in_dim):
            w = int(weights_i8[o, j])
            x = int(inputs[j])
            acc += x * w
        acc += int(biases[o])

        # Signed 16-bit wrap
        acc = (acc + 32768) % 65536 - 32768

        # Arithmetic right shift by 2
        acc >>= 2

        outputs[o] = np.int16(acc)

    return outputs


def generate_test_vectors(
    in_dim: int = 256,
    out_dim: int = 128,
    output_path: str = "test_vectors.json",
):
    rng = np.random.RandomState(42)

    # Random quantized weights in {-2, -1, 0, +1}
    weights_i8 = rng.randint(-2, 2, size=(out_dim, in_dim)).astype(np.int8)

    # Random biases in int16 range (-500 .. 500 to avoid extreme overflow)
    biases = rng.randint(-500, 501, size=out_dim).astype(np.int16)

    # Random inputs in Z80 domain (scaled by ×32 from float)
    # Simulate trigram bucket counts (0..100-ish)
    inputs = rng.randint(0, 101, size=in_dim).astype(np.int16)

    # Pack weights to match binary format
    packed_bytes = pack_2bit_weights(weights_i8)

    # Compute expected outputs
    expected = simulate_layer_python(weights_i8, biases, inputs)

    # Build JSON
    vectors = {
        "description": f"Layer {in_dim}→{out_dim}",
        "in_dim": in_dim,
        "out_dim": out_dim,
        "weights_base64": base64.b64encode(packed_bytes).decode("ascii"),
        "weights_packed_len": len(packed_bytes),
        "biases": [int(b) for b in biases],
        "inputs": [int(x) for x in inputs],
        "expected_outputs": [int(o) for o in expected],
    }

    with open(output_path, "w") as f:
        json.dump(vectors, f, indent=2)

    print(f"Generated {output_path}: {in_dim}→{out_dim}, "
          f"{len(packed_bytes)} packed weight bytes")
    return vectors


def verify_single_vector(vectors: dict):
    """Self-check: unpack weights and recompute, verify round-trip."""
    packed = base64.b64decode(vectors["weights_base64"])
    n_weights = vectors["in_dim"] * vectors["out_dim"]
    unpacked = unpack_2bit_weights(packed, n_weights)
    weights_i8 = unpacked.reshape(vectors["out_dim"], vectors["in_dim"])

    biases = np.array(vectors["biases"], dtype=np.int16)
    inputs = np.array(vectors["inputs"], dtype=np.int16)
    expected = np.array(vectors["expected_outputs"], dtype=np.int16)

    recomputed = simulate_layer_python(weights_i8, biases, inputs)

    diff = int(np.abs(recomputed.astype(np.int32) - expected.astype(np.int32)).max())
    ok = diff == 0
    print(f"Self-check: {vectors['description']}: {'PASS' if ok else 'FAIL'} (max diff: {diff})")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="test_vectors.json")
    parser.add_argument("--in-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=128)
    args = parser.parse_args()

    vecs = generate_test_vectors(args.in_dim, args.out_dim, args.output)
    verify_single_vector(vecs)
