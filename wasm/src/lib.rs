/// Tier 2.5 Wasm Kernel -- Float Transformer
///
/// Supports four weight formats:
///   - matmul_f32w:     fp32 weights (current production path)
///   - matmul_8bit:     i8 weights + per-tensor scale
///   - matmul_4bit:     4-bit packed weights + per-tensor scale
///   - matmul_ternary:  2-bit ternary weights (4 per byte) + per-tensor scale
///
/// 4-bit packing convention (matches py/serialize.py):
///   Each byte stores 2 weights as 4-bit offset-binary.
///   High nibble (bits 7-4) = first weight.  Low nibble (bits 3-0) = second weight.
///   Stored value = (int_weight + 8) & 0x0F, so int_weight in [-8, 7].
///   Unpack: ((nibble as i32) - 8) as f32 * scale.

// --- 4-bit packed quantized matmul ---
//
// Packing layout produced by py/serialize.py quantize_4bit():
//   byte = (hi_nibble << 4) | lo_nibble
//   hi_nibble = (quant[2k]   + 8) & 0xF   <- even index = HIGH nibble
//   lo_nibble = (quant[2k+1] + 8) & 0xF   <- odd  index = LOW  nibble

#[no_mangle]
pub unsafe extern "C" fn matmul_4bit(
    weights: *const u8,
    scale: f32,
    biases: *const f32,
    input: *const f32,
    output: *mut f32,
    in_dim: i32,
    out_dim: i32,
) {
    let in_dim = in_dim as usize;
    let out_dim = out_dim as usize;
    let packed_total = (in_dim * out_dim + 1) / 2;
    let weight_bytes = core::slice::from_raw_parts(weights, packed_total);
    let bias_slice = core::slice::from_raw_parts(biases, out_dim);
    let input_slice = core::slice::from_raw_parts(input, in_dim);
    let output_slice = core::slice::from_raw_parts_mut(output, out_dim);
    for o in 0..out_dim {
        let mut sum = bias_slice[o];
        for i in 0..in_dim {
            let flat_idx = o * in_dim + i;
            let byte = weight_bytes[flat_idx / 2];
            let nibble = if flat_idx % 2 == 0 { (byte >> 4) & 0x0F } else { byte & 0x0F };
            let w = (nibble as i32 - 8) as f32 * scale;
            sum += w * input_slice[i];
        }
        output_slice[o] = sum;
    }
}

// --- 4-bit per-group-scale matmul ---
// Same nibble packing as matmul_4bit, but one scale per group_size weights
// instead of one per tensor. Dramatically reduces quantization error on
// layers with mixed-magnitude weights.

#[no_mangle]
pub unsafe extern "C" fn matmul_4bit_grouped(
    weights: *const u8,
    scales: *const f32,   // array of per-group scales, len = ceil(in_dim*out_dim / group_size)
    biases: *const f32,
    input: *const f32,
    output: *mut f32,
    in_dim: i32,
    out_dim: i32,
    group_size: i32,
) {
    let in_dim = in_dim as usize;
    let out_dim = out_dim as usize;
    let group_size = group_size as usize;
    let total = in_dim * out_dim;
    let n_groups = (total + group_size - 1) / group_size;
    let packed_total = (total + 1) / 2;

    let weight_bytes = core::slice::from_raw_parts(weights, packed_total);
    let scales_slice = core::slice::from_raw_parts(scales, n_groups);
    let bias_slice   = core::slice::from_raw_parts(biases, out_dim);
    let input_slice  = core::slice::from_raw_parts(input, in_dim);
    let output_slice = core::slice::from_raw_parts_mut(output, out_dim);

    for o in 0..out_dim {
        let mut sum = bias_slice[o];
        for i in 0..in_dim {
            let flat_idx = o * in_dim + i;
            let byte = weight_bytes[flat_idx / 2];
            let nibble = if flat_idx % 2 == 0 { (byte >> 4) & 0x0F } else { byte & 0x0F };
            let scale = scales_slice[flat_idx / group_size];
            let w = (nibble as i32 - 8) as f32 * scale;
            sum += w * input_slice[i];
        }
        output_slice[o] = sum;
    }
}

// --- 8-bit signed quantized matmul ---

#[no_mangle]
pub unsafe extern "C" fn matmul_8bit(
    weights: *const i8,
    scale: f32,
    biases: *const f32,
    input: *const f32,
    output: *mut f32,
    in_dim: i32,
    out_dim: i32,
) {
    let in_dim = in_dim as usize;
    let out_dim = out_dim as usize;
    let weight_slice = core::slice::from_raw_parts(weights, out_dim * in_dim);
    let bias_slice = core::slice::from_raw_parts(biases, out_dim);
    let input_slice = core::slice::from_raw_parts(input, in_dim);
    let output_slice = core::slice::from_raw_parts_mut(output, out_dim);
    for o in 0..out_dim {
        let mut sum = bias_slice[o];
        for i in 0..in_dim {
            let w = weight_slice[o * in_dim + i] as f32 * scale;
            sum += w * input_slice[i];
        }
        output_slice[o] = sum;
    }
}

// --- Legacy dead 4-bit path (kept, not used in production) ---

const GLOBAL_WEIGHT_SCALE: f32 = 0.4;

#[inline(always)]
fn unpack_weight_f32_legacy(packed: &[u8], idx: usize) -> f32 {
    let byte = packed[idx / 2];
    let shift = ((idx % 2) * 4) as u32;
    let raw = ((byte >> shift) & 0x0F) as i32;
    (raw - 8) as f32 * GLOBAL_WEIGHT_SCALE
}

#[no_mangle]
pub unsafe extern "C" fn matmul_f32(
    weights: *const u8,
    biases: *const f32,
    input: *const f32,
    output: *mut f32,
    in_dim: i32,
    out_dim: i32,
) {
    let in_dim = in_dim as usize;
    let out_dim = out_dim as usize;
    let packed_total = (in_dim * out_dim + 1) / 2;
    let weight_bytes = core::slice::from_raw_parts(weights, packed_total);
    let bias_slice = core::slice::from_raw_parts(biases, out_dim);
    let input_slice = core::slice::from_raw_parts(input, in_dim);
    let output_slice = core::slice::from_raw_parts_mut(output, out_dim);
    for o in 0..out_dim {
        let mut sum = bias_slice[o];
        for i in 0..in_dim {
            let w = unpack_weight_f32_legacy(weight_bytes, o * in_dim + i);
            sum += w * input_slice[i];
        }
        output_slice[o] = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn matmul_no_bias_f32(
    weights: *const u8,
    input: *const f32,
    output: *mut f32,
    in_dim: i32,
    out_dim: i32,
) {
    let in_dim = in_dim as usize;
    let out_dim = out_dim as usize;
    let packed_total = (in_dim * out_dim + 1) / 2;
    let weight_bytes = core::slice::from_raw_parts(weights, packed_total);
    let input_slice = core::slice::from_raw_parts(input, in_dim);
    let output_slice = core::slice::from_raw_parts_mut(output, out_dim);
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            let w = unpack_weight_f32_legacy(weight_bytes, o * in_dim + i);
            sum += w * input_slice[i];
        }
        output_slice[o] = sum;
    }
}

// --- Softmax ---

#[no_mangle]
pub unsafe extern "C" fn softmax_f32(data: *mut f32, len: i32) {
    let len = len as usize;
    let slice = core::slice::from_raw_parts_mut(data, len);
    if len == 0 { return; }
    let mut max_val = f32::NEG_INFINITY;
    for v in slice.iter() { if *v > max_val { max_val = *v; } }
    let mut sum = 0.0f32;
    for v in slice.iter_mut() { *v = (*v - max_val).exp(); sum += *v; }
    if sum > 0.0 {
        for v in slice.iter_mut() { *v /= sum; }
    } else {
        let v = 1.0 / len as f32;
        for val in slice.iter_mut() { *val = v; }
    }
}

#[no_mangle]
pub unsafe extern "C" fn softmax_causal_f32(data: *mut f32, seq_len: i32) {
    let s = seq_len as usize;
    let slice = core::slice::from_raw_parts_mut(data, s * s);
    for t in 0..s {
        for i in (t + 1)..s { slice[t * s + i] = f32::NEG_INFINITY; }
    }
    for t in 0..s {
        let row_start = t * s;
        let row_end = row_start + s;
        let mut max_val = f32::NEG_INFINITY;
        for i in row_start..row_end { if slice[i] > max_val { max_val = slice[i]; } }
        let mut sum = 0.0f32;
        for i in row_start..row_end {
            if slice[i] == f32::NEG_INFINITY { slice[i] = 0.0; }
            else { slice[i] = (slice[i] - max_val).exp(); sum += slice[i]; }
        }
        if sum > 0.0 { for i in row_start..row_end { slice[i] /= sum; } }
    }
}

// --- Layer Normalization ---

#[no_mangle]
pub unsafe extern "C" fn layer_norm_f32(x: *mut f32, gamma: *const f32, beta: *const f32, dim: i32, eps: f32) {
    let d = dim as usize;
    let x_slice = core::slice::from_raw_parts_mut(x, d);
    let g = core::slice::from_raw_parts(gamma, d);
    let b = core::slice::from_raw_parts(beta, d);
    if d == 0 { return; }
    let mean: f32 = x_slice.iter().sum::<f32>() / d as f32;
    let var: f32 = x_slice.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / d as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..d { x_slice[i] = (x_slice[i] - mean) * inv_std * g[i] + b[i]; }
}

// --- RMSNorm ---
// Used by modern architecture (no mean subtraction, no bias).
// In-place: x = x / rms(x) * gamma

#[no_mangle]
pub unsafe extern "C" fn rms_norm_f32(x: *mut f32, gamma: *const f32, dim: i32, eps: f32) {
    let d = dim as usize;
    let x_slice = core::slice::from_raw_parts_mut(x, d);
    let g = core::slice::from_raw_parts(gamma, d);
    if d == 0 { return; }
    let rms_sq: f32 = x_slice.iter().map(|v| v * v).sum::<f32>() / d as f32;
    let inv_rms = 1.0 / (rms_sq + eps).sqrt();
    for i in 0..d { x_slice[i] = x_slice[i] * inv_rms * g[i]; }
}

// --- SiLU activation (Swish) ---
// In-place: x = x * sigmoid(x) = x / (1 + exp(-x))

#[no_mangle]
pub unsafe extern "C" fn silu_f32(data: *mut f32, len: i32) {
    let slice = core::slice::from_raw_parts_mut(data, len as usize);
    for v in slice.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

// --- Element-wise multiply: a[i] *= b[i] ---

#[no_mangle]
pub unsafe extern "C" fn mul_vec_f32(a: *mut f32, b: *const f32, len: i32) {
    let a_slice = core::slice::from_raw_parts_mut(a, len as usize);
    let b_slice = core::slice::from_raw_parts(b, len as usize);
    for i in 0..len as usize { a_slice[i] *= b_slice[i]; }
}

// --- Vector operations ---

#[no_mangle]
pub unsafe extern "C" fn add_vec_f32(a: *mut f32, b: *const f32, len: i32) {
    let a_slice = core::slice::from_raw_parts_mut(a, len as usize);
    let b_slice = core::slice::from_raw_parts(b, len as usize);
    for i in 0..len as usize { a_slice[i] += b_slice[i]; }
}

#[no_mangle]
pub unsafe extern "C" fn relu_f32(data: *mut f32, len: i32) {
    let slice = core::slice::from_raw_parts_mut(data, len as usize);
    for v in slice.iter_mut() { if *v < 0.0 { *v = 0.0; } }
}

#[no_mangle]
pub unsafe extern "C" fn scale_f32(data: *mut f32, len: i32, scale: f32) {
    let slice = core::slice::from_raw_parts_mut(data, len as usize);
    for v in slice.iter_mut() { *v *= scale; }
}

#[no_mangle]
pub unsafe extern "C" fn matmul_f32w(
    weights: *const f32,
    biases: *const f32,
    input: *const f32,
    output: *mut f32,
    in_dim: i32,
    out_dim: i32,
) {
    let in_dim = in_dim as usize;
    let out_dim = out_dim as usize;
    let w = core::slice::from_raw_parts(weights, out_dim * in_dim);
    let b = core::slice::from_raw_parts(biases, out_dim);
    let inp = core::slice::from_raw_parts(input, in_dim);
    let out = core::slice::from_raw_parts_mut(output, out_dim);
    for o in 0..out_dim {
        let mut sum = b[o];
        for i in 0..in_dim { sum += w[o * in_dim + i] * inp[i]; }
        out[o] = sum;
    }
}

/// BF16 (bfloat16) matmul. Weights stored as u16; bf16 is the top 16 bits of
/// the f32 bit pattern, so conversion is a single shift — no scale needed.
/// Precision loss is minimal (~3 decimal digits vs ~7 for fp32).
#[no_mangle]
pub unsafe extern "C" fn matmul_bf16(
    weights: *const u16,
    biases:  *const f32,
    input:   *const f32,
    output:  *mut f32,
    in_dim:  i32,
    out_dim: i32,
) {
    let in_dim  = in_dim  as usize;
    let out_dim = out_dim as usize;
    let w   = core::slice::from_raw_parts(weights, out_dim * in_dim);
    let b   = core::slice::from_raw_parts(biases, out_dim);
    let inp = core::slice::from_raw_parts(input, in_dim);
    let out = core::slice::from_raw_parts_mut(output, out_dim);
    for o in 0..out_dim {
        let mut sum = b[o];
        for i in 0..in_dim {
            // bf16 → f32: place the u16 bits in the high half of a u32
            let wf = f32::from_bits((w[o * in_dim + i] as u32) << 16);
            sum += wf * inp[i];
        }
        out[o] = sum;
    }
}

/// Multi-head causal self-attention (float32).
///
/// qkv:      [seq, d*3]  — Q | K | V interleaved per position
///           qkv[p*d*3 + 0..d]    = Q[p]
///           qkv[p*d*3 + d..2d]   = K[p]
///           qkv[p*d*3 + 2d..3d]  = V[p]
/// scores:   [seq, seq] scratch buffer (overwritten)
/// attn_out: [seq, d]   output
#[no_mangle]
pub unsafe extern "C" fn attention_f32(
    qkv:      *const f32,
    scores:   *mut f32,
    attn_out: *mut f32,
    seq:      i32,
    d:        i32,
    n_heads:  i32,
) {
    let seq = seq as usize;
    let d   = d   as usize;
    let nh  = n_heads as usize;
    let dh  = d / nh;
    let inv_sqrt_dh = 1.0_f32 / (dh as f32).sqrt();

    let qkv_s  = core::slice::from_raw_parts(qkv, seq * d * 3);
    let sc     = core::slice::from_raw_parts_mut(scores, seq * seq);
    let out    = core::slice::from_raw_parts_mut(attn_out, seq * d);

    // Zero output buffer — heads accumulate into it.
    for v in out.iter_mut() { *v = 0.0; }

    for h in 0..nh {
        let ho = h * dh; // head offset within d

        // 1. Q·Kᵀ with causal mask → scores[seq, seq]
        for qi in 0..seq {
            for kj in 0..=qi {
                let mut dot = 0.0_f32;
                for x in 0..dh {
                    dot += qkv_s[qi * d * 3 + ho + x]
                         * qkv_s[kj * d * 3 + d + ho + x];
                }
                sc[qi * seq + kj] = dot * inv_sqrt_dh;
            }
            for kj in (qi + 1)..seq { sc[qi * seq + kj] = f32::NEG_INFINITY; }
        }

        // 2. Softmax per row (causal — upper triangle already -inf)
        for qi in 0..seq {
            let row = qi * seq;
            let mut max_val = f32::NEG_INFINITY;
            for kj in 0..=qi { if sc[row + kj] > max_val { max_val = sc[row + kj]; } }
            let mut sum = 0.0_f32;
            for kj in 0..=qi { sc[row + kj] = (sc[row + kj] - max_val).exp(); sum += sc[row + kj]; }
            if sum > 0.0 { for kj in 0..=qi { sc[row + kj] /= sum; } }
            for kj in (qi + 1)..seq { sc[row + kj] = 0.0; }
        }

        // 3. Weighted V sum → attn_out[seq, d]
        for qi in 0..seq {
            for x in 0..dh {
                let mut val = 0.0_f32;
                for kj in 0..=qi {
                    val += sc[qi * seq + kj] * qkv_s[kj * d * 3 + d * 2 + ho + x];
                }
                out[qi * d + ho + x] = val;
            }
        }
    }
}

// --- SIMD128 matmul ---
//
// matmul_f32w_simd: same semantics as matmul_f32w, implemented with Wasm SIMD128.
// Processes 4 floats per instruction (f32x4). For dimensions not divisible by 4,
// a scalar tail loop handles the remainder.
//
// Requires the Wasm SIMD128 feature at build time:
//   RUSTFLAGS='-C target-feature=+simd128' cargo build ...
//
// The export name is different from matmul_f32w so both can coexist in the same
// binary: JS detects the SIMD export and uses it when present, falling back to
// the scalar path for older browsers.
#[cfg(target_feature = "simd128")]
#[no_mangle]
pub unsafe extern "C" fn matmul_f32w_simd(
    weights: *const f32,
    biases:  *const f32,
    input:   *const f32,
    output:  *mut f32,
    in_dim:  i32,
    out_dim: i32,
) {
    use core::arch::wasm32::*;
    let in_dim  = in_dim  as usize;
    let out_dim = out_dim as usize;
    let w   = core::slice::from_raw_parts(weights, out_dim * in_dim);
    let b   = core::slice::from_raw_parts(biases, out_dim);
    let inp = core::slice::from_raw_parts(input, in_dim);
    let out = core::slice::from_raw_parts_mut(output, out_dim);

    let chunks = in_dim / 4;
    let tail   = in_dim % 4;

    for o in 0..out_dim {
        let w_row = &w[o * in_dim..];
        let mut acc = f32x4_splat(0.0_f32);

        for i in 0..chunks {
            let wv = v128_load(w_row.as_ptr().add(i * 4) as *const v128);
            let iv = v128_load(inp.as_ptr().add(i * 4) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(wv, iv));
        }

        // Horizontal sum of the four SIMD lanes
        let mut sum = b[o]
            + f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        // Scalar tail for the remaining 0-3 elements
        let base = chunks * 4;
        for i in 0..tail {
            sum += w_row[base + i] * inp[base + i];
        }

        out[o] = sum;
    }
}

// --- Ternary matmul ---
//
// Packing layout produced by py/serialize.py quantize_ternary():
//   4 ternary codes per byte, high bits first (2 bits each)
//   code 0b00 (-1): negative  → subtract scale * input[i]
//   code 0b01 ( 0): zero      → no-op
//   code 0b10 (+1): positive  → add scale * input[i]
//   code 0b11:      unused, treated as zero

#[no_mangle]
pub unsafe extern "C" fn matmul_ternary(
    weights: *const u8,
    scale: f32,
    biases: *const f32,
    input: *const f32,
    output: *mut f32,
    in_dim: i32,
    out_dim: i32,
) {
    let in_dim  = in_dim  as usize;
    let out_dim = out_dim as usize;
    let n_bytes = (in_dim * out_dim + 3) / 4;
    let weight_bytes = core::slice::from_raw_parts(weights, n_bytes);
    let bias_slice   = core::slice::from_raw_parts(biases,  out_dim);
    let input_slice  = core::slice::from_raw_parts(input,   in_dim);
    let output_slice = core::slice::from_raw_parts_mut(output, out_dim);

    for o in 0..out_dim {
        let mut sum = bias_slice[o];
        for i in 0..in_dim {
            let flat     = o * in_dim + i;
            let byte_idx = flat / 4;
            let shift    = 6 - (flat % 4) * 2;   // high bits first
            let code     = (weight_bytes[byte_idx] >> shift) & 0x3;
            match code {
                0 => sum -= scale * input_slice[i],  // -1
                2 => sum += scale * input_slice[i],  // +1
                _ => {}                               //  0 (codes 1 and 3)
            }
        }
        output_slice[o] = sum;
    }
}
