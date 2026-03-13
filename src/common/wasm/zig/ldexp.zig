//! WASM element-wise ldexp kernels: out[i] = x1[i] * 2^x2.
//!
//! Scalar variant: x2 is a single integer, pre-computed as multiplier.
//! Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");
const math = @import("std").math;

/// ldexp for f64 (scalar exponent): out[i] = x1[i] * 2^exp.
/// Pre-computes the multiplier for efficiency.
export fn ldexp_scalar_f64(x1: [*]const f64, out: [*]f64, N: u32, exp: f64) void {
    const multiplier = math.pow(f64, 2.0, exp);
    const vmul: simd.V2f64 = @splat(multiplier);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(x1, i);
        simd.store2_f64(out, i, v * vmul);
    }
    while (i < N) : (i += 1) {
        out[i] = x1[i] * multiplier;
    }
}

/// ldexp for f32 (scalar exponent): out[i] = x1[i] * 2^exp.
export fn ldexp_scalar_f32(x1: [*]const f32, out: [*]f32, N: u32, exp: f32) void {
    const multiplier = math.pow(f32, 2.0, exp);
    const vmul: simd.V4f32 = @splat(multiplier);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(x1, i);
        simd.store4_f32(out, i, v * vmul);
    }
    while (i < N) : (i += 1) {
        out[i] = x1[i] * multiplier;
    }
}

// --- Tests ---

test "ldexp_scalar_f64 basic" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ 1.0, 2.0, 0.5, -1.0, 3.0 };
    var out: [5]f64 = undefined;
    ldexp_scalar_f64(&x1, &out, 5, 3.0); // multiply by 8
    try testing.expectApproxEqAbs(out[0], 8.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 16.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], -8.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 24.0, 1e-10);
}
