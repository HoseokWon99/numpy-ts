//! WASM inner-product kernels for all numeric types.
//!
//! Convention: C = inner(A, B) where A is (M x K), B is (N x K), C is (M x N).
//! C[i,j] = sum_k A[i*K+k] * B[j*K+k]   (dot product of row i of A with row j of B)
//!
//! All matrices are row-major (C-contiguous).
//! Complex matrices are interleaved [re, im, re, im, ...]; M, N, K are element counts.
//!
//! Unlike matmul, both A and B are accessed row-wise along K, so the memory access
//! pattern is inherently cache-friendly — no transpose or tiling needed.

const simd = @import("simd.zig");

/// Computes C = inner(A, B) for row-major f64 arrays.
/// A is (M x K), B is (N x K), C is (M x N).
export fn inner_f64(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2f64 (2-wide)

    for (0..M) |i| {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V2f64 = @splat(0);

            // SIMD loop: 2 f64s per iteration
            var k: u32 = 0;
            while (k < k_simd) : (k += 2) {
                acc += simd.load2_f64(a, a_row + k) * simd.load2_f64(b, b_row + k);
            }

            // Horizontal sum + scalar remainder
            var sum: f64 = acc[0] + acc[1];
            while (k < K) : (k += 1) {
                sum += a[a_row + k] * b[b_row + k];
            }

            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major f32 arrays.
/// A is (M x K), B is (N x K), C is (M x N).
export fn inner_f32(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4f32 (4-wide)

    for (0..M) |i| {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V4f32 = @splat(0);

            // SIMD loop: 4 f32s per iteration
            var k: u32 = 0;
            while (k < k_simd) : (k += 4) {
                acc += simd.load4_f32(a, a_row + k) * simd.load4_f32(b, b_row + k);
            }

            // Horizontal sum + scalar remainder
            var sum: f32 = acc[0] + acc[1] + acc[2] + acc[3];
            while (k < K) : (k += 1) {
                sum += a[a_row + k] * b[b_row + k];
            }

            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major complex128 arrays.
/// Data is interleaved f64 [re, im, re, im, ...].
/// M, N, K are element counts (each element = 2 f64s).
/// C[i,j] = sum_k A[i,k] * B[j,k]  (NumPy inner does NOT conjugate)
export fn inner_c128(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {
    for (0..M) |i| {
        const a_row = i * K * 2; // 2 f64s per complex element
        for (0..N) |j| {
            const b_row = j * K * 2;
            var sum_re: f64 = 0;
            var sum_im: f64 = 0;

            for (0..K) |k| {
                const ak = k * 2;
                const a_re = a[a_row + ak];
                const a_im = a[a_row + ak + 1];
                const b_re = b[b_row + ak];
                const b_im = b[b_row + ak + 1];
                // (a_re + a_im*i) * (b_re + b_im*i)
                sum_re += a_re * b_re - a_im * b_im;
                sum_im += a_re * b_im + a_im * b_re;
            }

            const out_idx = (i * N + j) * 2;
            c[out_idx] = sum_re;
            c[out_idx + 1] = sum_im;
        }
    }
}

/// Computes C = inner(A, B) for row-major complex64 arrays.
/// Data is interleaved f32 [re, im, re, im, ...].
/// M, N, K are element counts (each element = 2 f32s).
export fn inner_c64(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {
    for (0..M) |i| {
        const a_row = i * K * 2; // 2 f32s per complex element
        for (0..N) |j| {
            const b_row = j * K * 2;
            var sum_re: f32 = 0;
            var sum_im: f32 = 0;

            for (0..K) |k| {
                const ak = k * 2;
                const a_re = a[a_row + ak];
                const a_im = a[a_row + ak + 1];
                const b_re = b[b_row + ak];
                const b_im = b[b_row + ak + 1];
                // (a_re + a_im*i) * (b_re + b_im*i)
                sum_re += a_re * b_re - a_im * b_im;
                sum_im += a_re * b_im + a_im * b_re;
            }

            const out_idx = (i * N + j) * 2;
            c[out_idx] = sum_re;
            c[out_idx + 1] = sum_im;
        }
    }
}

/// Computes C = inner(A, B) for row-major i64 arrays with wrapping arithmetic.
/// Handles both signed (i64) and unsigned (u64).
export fn inner_i64(a: [*]const i64, b: [*]const i64, c: [*]i64, M: u32, N: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2i64 (2-wide)

    for (0..M) |i| {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V2i64 = @splat(0);

            // SIMD loop: 2 i64s per iteration
            var k: u32 = 0;
            while (k < k_simd) : (k += 2) {
                acc +%= simd.load2_i64(a, a_row + k) *% simd.load2_i64(b, b_row + k);
            }

            // Horizontal sum + scalar remainder
            var sum: i64 = acc[0] +% acc[1];
            while (k < K) : (k += 1) {
                sum +%= a[a_row + k] *% b[b_row + k];
            }

            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major i32 arrays with wrapping arithmetic.
/// Handles both signed (i32) and unsigned (u32).
export fn inner_i32(a: [*]const i32, b: [*]const i32, c: [*]i32, M: u32, N: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4i32 (4-wide)

    for (0..M) |i| {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V4i32 = @splat(0);

            // SIMD loop: 4 i32s per iteration
            var k: u32 = 0;
            while (k < k_simd) : (k += 4) {
                acc +%= simd.load4_i32(a, a_row + k) *% simd.load4_i32(b, b_row + k);
            }

            // Horizontal sum + scalar remainder
            var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
            while (k < K) : (k += 1) {
                sum +%= a[a_row + k] *% b[b_row + k];
            }

            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major i16 arrays with wrapping arithmetic.
/// Handles both signed (i16) and unsigned (u16).
export fn inner_i16(a: [*]const i16, b: [*]const i16, c: [*]i16, M: u32, N: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 7); // floor to V8i16 (8-wide)

    for (0..M) |i| {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V8i16 = @splat(0);

            // SIMD loop: 8 i16s per iteration
            var k: u32 = 0;
            while (k < k_simd) : (k += 8) {
                acc +%= simd.load8_i16(a, a_row + k) *% simd.load8_i16(b, b_row + k);
            }

            // Horizontal sum + scalar remainder
            var sum: i16 = acc[0] +% acc[1] +% acc[2] +% acc[3] +% acc[4] +% acc[5] +% acc[6] +% acc[7];
            while (k < K) : (k += 1) {
                sum +%= a[a_row + k] *% b[b_row + k];
            }

            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major i8 arrays with wrapping arithmetic.
/// Handles both signed (i8) and unsigned (u8).
/// Uses widened i16 multiply via simd.muladd_i8x16 (WASM has no native i8x16.mul).
export fn inner_i8(a: [*]const i8, b: [*]const i8, c: [*]i8, M: u32, N: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 15); // floor to V16i8 (16-wide)

    for (0..M) |i| {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V16i8 = @splat(0);

            // SIMD loop: 16 i8s per iteration (widened i16 multiply via muladd)
            var k: u32 = 0;
            while (k < k_simd) : (k += 16) {
                const a_load = simd.load16_i8(a, a_row + k);
                const b_load = simd.load16_i8(b, b_row + k);
                acc = simd.muladd_i8x16(acc, a_load, b_load);
            }

            // Horizontal sum of 16 lanes + scalar remainder
            var sum: i8 = 0;
            for (0..16) |lane| {
                sum +%= acc[lane];
            }
            while (k < K) : (k += 1) {
                sum +%= a[a_row + k] *% b[b_row + k];
            }

            c[i * N + j] = sum;
        }
    }
}

// --- Tests ---

test "inner_f64 2x3 @ 2x3 → 2x2" {
    const testing = @import("std").testing;
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8, 9], [10, 11, 12]]
    // [0,0] = 1*7 + 2*8 + 3*9 = 50
    // [0,1] = 1*10 + 2*11 + 3*12 = 68
    // [1,0] = 4*7 + 5*8 + 6*9 = 122
    // [1,1] = 4*10 + 5*11 + 6*12 = 167
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f64{ 7, 8, 9, 10, 11, 12 };
    var c: [4]f64 = undefined;
    inner_f64(&a, &b, &c, 2, 2, 3);
    try testing.expectApproxEqAbs(c[0], 50.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 68.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 122.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 167.0, 1e-10);
}

test "inner_f32 3x2 @ 2x2 → 3x2" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 0, 0, 1 };
    var c: [6]f32 = undefined;
    inner_f32(&a, &b, &c, 3, 2, 2);
    try testing.expectApproxEqAbs(c[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 3.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 4.0, 1e-5);
    try testing.expectApproxEqAbs(c[4], 5.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 6.0, 1e-5);
}

test "inner_c128 1x2 @ 1x2 → 1x1" {
    const testing = @import("std").testing;
    // A = [(1+2i), (3+4i)], B = [(5+6i), (7+8i)]
    // inner = (1+2i)*(5+6i) + (3+4i)*(7+8i) = -18+68i
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 5, 6, 7, 8 };
    var c: [2]f64 = undefined;
    inner_c128(&a, &b, &c, 1, 1, 2);
    try testing.expectApproxEqAbs(c[0], -18.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 68.0, 1e-10);
}

test "inner_c64 1x2 @ 1x2 → 1x1" {
    const testing = @import("std").testing;
    // A = [(1+2i), (3+4i)], B = [(5+6i), (7+8i)]
    // inner = (1+2i)*(5+6i) + (3+4i)*(7+8i) = -18+68i
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c: [2]f32 = undefined;
    inner_c64(&a, &b, &c, 1, 1, 2);
    try testing.expectApproxEqAbs(c[0], -18.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 68.0, 1e-5);
}

test "inner_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3 };
    const b = [_]i64{ 4, 5, 6 };
    var c: [1]i64 = undefined;
    inner_i64(&a, &b, &c, 1, 1, 3);
    try testing.expectEqual(c[0], 32); // 1*4 + 2*5 + 3*6 = 32
}

test "inner_i32 2x3 @ 2x3 → 2x2" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i32{ 7, 8, 9, 10, 11, 12 };
    var c: [4]i32 = undefined;
    inner_i32(&a, &b, &c, 2, 2, 3);
    try testing.expectEqual(c[0], 50);
    try testing.expectEqual(c[1], 68);
    try testing.expectEqual(c[2], 122);
    try testing.expectEqual(c[3], 167);
}

test "inner_i16 wrapping" {
    const testing = @import("std").testing;
    // Values that will overflow i16 when accumulated
    const a = [_]i16{ 127, 127, 127, 127 };
    const b = [_]i16{ 127, 127, 127, 127 };
    var c: [1]i16 = undefined;
    inner_i16(&a, &b, &c, 1, 1, 4);
    // 127*127*4 = 64516 → wraps in i16 to -1020
    const expected: i16 = @truncate(@as(i32, 127) * 127 * 4);
    try testing.expectEqual(c[0], expected);
}

test "inner_i8 wrapping" {
    const testing = @import("std").testing;
    const a = [_]i8{ 10, 10, 10, 10 };
    const b = [_]i8{ 10, 10, 10, 10 };
    var c: [1]i8 = undefined;
    inner_i8(&a, &b, &c, 1, 1, 4);
    // 10*10*4 = 400 → wraps in i8 to -112
    const expected: i8 = @truncate(@as(i32, 10) * 10 * 4);
    try testing.expectEqual(c[0], expected);
}
