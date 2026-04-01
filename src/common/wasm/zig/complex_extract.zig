//! Stride-2 extraction kernels for complex real/imag parts.
//!
//! Complex data is interleaved: [re0, im0, re1, im1, ...]
//! These kernels extract the real or imaginary lane into a contiguous output.
//! N = number of complex elements (output length).

const simd = @import("simd.zig");

/// Extract real parts from complex128 (f64 pairs): out[i] = src[2*i].
export fn extract_real_f64(src: [*]const f64, out: [*]f64, N: u32) void {
    // Load 2 f64s = 1 complex element, take lane 0
    var i: u32 = 0;
    while (i + 2 <= N) : (i += 2) {
        const v0 = simd.load2_f64(src, i * 2);
        const v1 = simd.load2_f64(src, i * 2 + 2);
        // v0 = [re0, im0], v1 = [re1, im1] → want [re0, re1]
        const re = @shuffle(f64, v0, v1, [2]i32{ 0, -1 });
        simd.store2_f64(out, i, re);
    }
    while (i < N) : (i += 1) {
        out[i] = src[i * 2];
    }
}

/// Extract imaginary parts from complex128: out[i] = src[2*i + 1].
export fn extract_imag_f64(src: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i + 2 <= N) : (i += 2) {
        const v0 = simd.load2_f64(src, i * 2);
        const v1 = simd.load2_f64(src, i * 2 + 2);
        // v0 = [re0, im0], v1 = [re1, im1] → want [im0, im1]
        const im = @shuffle(f64, v0, v1, [2]i32{ 1, -2 });
        simd.store2_f64(out, i, im);
    }
    while (i < N) : (i += 1) {
        out[i] = src[i * 2 + 1];
    }
}

/// Extract real parts from complex64 (f32 pairs): out[i] = src[2*i].
export fn extract_real_f32(src: [*]const f32, out: [*]f32, N: u32) void {
    // Load 4 f32s = 2 complex elements, shuffle to extract real lanes
    var i: u32 = 0;
    while (i + 4 <= N) : (i += 4) {
        const v0 = simd.load4_f32(src, i * 2); // [re0, im0, re1, im1]
        const v1 = simd.load4_f32(src, i * 2 + 4); // [re2, im2, re3, im3]
        // want [re0, re1, re2, re3]
        const re = @shuffle(f32, v0, v1, [4]i32{ 0, 2, -1, -3 });
        simd.store4_f32(out, i, re);
    }
    while (i < N) : (i += 1) {
        out[i] = src[i * 2];
    }
}

/// Extract imaginary parts from complex64: out[i] = src[2*i + 1].
export fn extract_imag_f32(src: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i + 4 <= N) : (i += 4) {
        const v0 = simd.load4_f32(src, i * 2);
        const v1 = simd.load4_f32(src, i * 2 + 4);
        // want [im0, im1, im2, im3]
        const im = @shuffle(f32, v0, v1, [4]i32{ 1, 3, -2, -4 });
        simd.store4_f32(out, i, im);
    }
    while (i < N) : (i += 1) {
        out[i] = src[i * 2 + 1];
    }
}

// --- Tests ---

test "extract_real_f64 basic" {
    const testing = @import("std").testing;
    // [re0, im0, re1, im1, re2, im2]
    const src = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var out: [3]f64 = undefined;
    extract_real_f64(&src, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 5.0, 1e-10);
}

test "extract_imag_f64 basic" {
    const testing = @import("std").testing;
    const src = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var out: [3]f64 = undefined;
    extract_imag_f64(&src, &out, 3);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 6.0, 1e-10);
}

test "extract_real_f32 basic" {
    const testing = @import("std").testing;
    const src = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var out: [5]f32 = undefined;
    extract_real_f32(&src, &out, 5);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 7.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 9.0, 1e-5);
}

test "extract_imag_f32 basic" {
    const testing = @import("std").testing;
    const src = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var out: [5]f32 = undefined;
    extract_imag_f32(&src, &out, 5);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 6.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 8.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 10.0, 1e-5);
}
