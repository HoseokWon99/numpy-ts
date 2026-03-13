//! WASM element-wise heaviside step function kernels for numeric types.
//!
//! Scalar: out[i] = x1[i] < 0 ? 0 : x1[i] == 0 ? x2 : 1
//! Binary: out[i] = x1[i] < 0 ? 0 : x1[i] == 0 ? x2[i] : 1
//! Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Heaviside step function for f64 (scalar x2): out[i] = x1[i]<0?0 : x1[i]==0?x2 : 1.
export fn heaviside_scalar_f64(x1: [*]const f64, out: [*]f64, N: u32, x2: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = x1[i];
        out[i] = if (v < 0.0) 0.0 else if (v == 0.0) x2 else 1.0;
    }
}

/// Heaviside step function for f32 (scalar x2): out[i] = x1[i]<0?0 : x1[i]==0?x2 : 1.
export fn heaviside_scalar_f32(x1: [*]const f32, out: [*]f32, N: u32, x2: f32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = x1[i];
        out[i] = if (v < 0.0) 0.0 else if (v == 0.0) x2 else 1.0;
    }
}

/// Heaviside step function for f64 (binary x2): out[i] = x1[i]<0?0 : x1[i]==0?x2[i] : 1.
export fn heaviside_f64(x1: [*]const f64, x2: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = x1[i];
        out[i] = if (v < 0.0) 0.0 else if (v == 0.0) x2[i] else 1.0;
    }
}

/// Heaviside step function for f32 (binary x2): out[i] = x1[i]<0?0 : x1[i]==0?x2[i] : 1.
export fn heaviside_f32(x1: [*]const f32, x2: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = x1[i];
        out[i] = if (v < 0.0) 0.0 else if (v == 0.0) x2[i] else 1.0;
    }
}

// --- Tests ---

test "heaviside_scalar_f64 basic" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ -2, -1, 0, 1, 2 };
    var out: [5]f64 = undefined;
    heaviside_scalar_f64(&x1, &out, 5, 0.5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.5, 1e-10);
    try testing.expectApproxEqAbs(out[3], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 1.0, 1e-10);
}

test "heaviside_f64 binary" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ -1, 0, 0, 1 };
    const x2 = [_]f64{ 99, 0.5, 0.7, 99 };
    var out: [4]f64 = undefined;
    heaviside_f64(&x1, &x2, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.5, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.7, 1e-10);
    try testing.expectApproxEqAbs(out[3], 1.0, 1e-10);
}

test "heaviside_scalar_f32 basic" {
    const testing = @import("std").testing;
    const x1 = [_]f32{ -2, -1, 0, 1, 2 };
    var out: [5]f32 = undefined;
    heaviside_scalar_f32(&x1, &out, 5, 0.5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 0.5, 1e-5);
    try testing.expectApproxEqAbs(out[3], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 1.0, 1e-5);
}

test "heaviside_f32 binary" {
    const testing = @import("std").testing;
    const x1 = [_]f32{ -1, 0, 0, 1 };
    const x2 = [_]f32{ 99, 0.5, 0.7, 99 };
    var out: [4]f32 = undefined;
    heaviside_f32(&x1, &x2, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.5, 1e-5);
    try testing.expectApproxEqAbs(out[2], 0.7, 1e-5);
    try testing.expectApproxEqAbs(out[3], 1.0, 1e-5);
}

test "heaviside_scalar_f64 x2=0" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ -1, 0, 1 };
    var out: [3]f64 = undefined;
    heaviside_scalar_f64(&x1, &out, 3, 0.0);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
}

test "heaviside_scalar_f64 x2=1" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ -1, 0, 1 };
    var out: [3]f64 = undefined;
    heaviside_scalar_f64(&x1, &out, 3, 1.0);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
}
