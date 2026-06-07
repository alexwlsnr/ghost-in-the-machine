/// Tier 2.5 Wasm Kernel — Float Transformer
///
/// 4-bit packed weights (space-efficient), float32 computation (correct).
/// This eliminates all integer scaling issues at the cost of slightly
/// larger kernel size. Weights are stored as packed int8 but converted
/// to float during computation using a global scale factor.

use core::arch::wasm32;

// ─── Weight unpacking ──────────────────────────────────────────────

const GLOBAL_WEIGHT_SCALE: f32 = 0.4;

#[inline(always)]
fn unpack_weight_f32(packed: &[u8], idx: usize) -> f32 {
    let byte = packed[idx / 2];
    let shift = ((idx % 2) * 4) as u32;
    let raw = ((byte >> shift) & 0x0F) as i32;
    (raw - 8) as f32 * GLOBAL_WEIGHT_SCALE
}

// ─── Core Matmul ───────────────────────────────────────────────────

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
            let w = unpack_weight_f32(weight_bytes, o * in_dim + i);
            sum += w * input_slice[i];
        }
        output_slice[o] = sum;
    }
}

/// Matmul without bias (for attention scores)
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
            let w = unpack_weight_f32(weight_bytes, o * in_dim + i);
            sum += w * input_slice[i];
        }
        output_slice[o] = sum;
    }
}

// ─── Softmax ─────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn softmax_f32(data: *mut f32, len: i32) {
    let len = len as usize;
    let slice = core::slice::from_raw_parts_mut(data, len);
    if len == 0 { return; }

    let mut max_val = f32::NEG_INFINITY;
    for v in slice.iter() {
        if *v > max_val { max_val = *v; }
    }

    let mut sum = 0.0f32;
    for v in slice.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    if sum > 0.0 {
        for v in slice.iter_mut() {
            *v /= sum;
        }
    } else {
        let v = 1.0 / len as f32;
        for val in slice.iter_mut() {
            *val = v;
        }
    }
}

/// Causal softmax: scores[seq_len, seq_len], upper triangle → 0
#[no_mangle]
pub unsafe extern "C" fn softmax_causal_f32(data: *mut f32, seq_len: i32) {
    let s = seq_len as usize;
    let slice = core::slice::from_raw_parts_mut(data, s * s);

    // Apply causal mask
    for t in 0..s {
        for i in (t + 1)..s {
            slice[t * s + i] = f32::NEG_INFINITY;
        }
    }

    // Softmax per row
    for t in 0..s {
        let row_start = t * s;
        let row_end = row_start + s;

        let mut max_val = f32::NEG_INFINITY;
        for i in row_start..row_end {
            if slice[i] > max_val { max_val = slice[i]; }
        }

        let mut sum = 0.0f32;
        for i in row_start..row_end {
            if slice[i] == f32::NEG_INFINITY {
                slice[i] = 0.0;
            } else {
                slice[i] = (slice[i] - max_val).exp();
                sum += slice[i];
            }
        }

        if sum > 0.0 {
            for i in row_start..row_end {
                slice[i] /= sum;
            }
        }
    }
}

// ─── Layer Normalization ─────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn layer_norm_f32(
    x: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    dim: i32,
    eps: f32,
) {
    let d = dim as usize;
    let x_slice = core::slice::from_raw_parts_mut(x, d);
    let g = core::slice::from_raw_parts(gamma, d);
    let b = core::slice::from_raw_parts(beta, d);
    if d == 0 { return; }

    // Mean
    let mean: f32 = x_slice.iter().sum::<f32>() / d as f32;

    // Variance
    let var: f32 = x_slice.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / d as f32;

    let inv_std = 1.0 / (var + eps).sqrt();

    for i in 0..d {
        x_slice[i] = (x_slice[i] - mean) * inv_std * g[i] + b[i];
    }
}

// ─── Vector operations ───────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn add_vec_f32(a: *mut f32, b: *const f32, len: i32) {
    let a_slice = core::slice::from_raw_parts_mut(a, len as usize);
    let b_slice = core::slice::from_raw_parts(b, len as usize);
    for i in 0..len as usize {
        a_slice[i] += b_slice[i];
    }
}

#[no_mangle]
pub unsafe extern "C" fn relu_f32(data: *mut f32, len: i32) {
    let slice = core::slice::from_raw_parts_mut(data, len as usize);
    for v in slice.iter_mut() {
        if *v < 0.0 { *v = 0.0; }
    }
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
        for i in 0..in_dim {
            sum += w[o * in_dim + i] * inp[i];
        }
        out[o] = sum;
    }
}

