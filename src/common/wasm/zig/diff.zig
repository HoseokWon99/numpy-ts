//! WASM diff kernels: out[i] = a[i+1] - a[i].
//!
//! 1D: flat diff of N elements.
//! 2D: per-row diff along last axis for contiguous [rows x cols] layout.

const simd = @import("simd.zig");

// ---- 1D diff ----

/// 1D diff for f64: out[i] = a[i+1] - a[i], N = output length.
export fn diff_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i + 1) - simd.load2_f64(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for f32.
export fn diff_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i + 1) - simd.load4_f32(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for i64 (scalar).
export fn diff_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for i32.
export fn diff_i32(a: [*]const i32, out: [*]i32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, simd.load4_i32(a, i + 1) - simd.load4_i32(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for i16.
export fn diff_i16(a: [*]const i16, out: [*]i16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, simd.load8_i16(a, i + 1) - simd.load8_i16(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for i8.
export fn diff_i8(a: [*]const i8, out: [*]i8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, simd.load16_i8(a, i + 1) - simd.load16_i8(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

// ---- 2D diff (per-row along last axis) ----

/// 2D diff for f64: per-row diff on [rows x cols] → [rows x (cols-1)].
export fn diff_2d_f64(a: [*]const f64, out: [*]f64, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 1);
        var i: u32 = 0;
        while (i < n_simd) : (i += 2) {
            simd.store2_f64(dst, i, simd.load2_f64(src, i + 1) - simd.load2_f64(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for f32.
export fn diff_2d_f32(a: [*]const f32, out: [*]f32, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 3);
        var i: u32 = 0;
        while (i < n_simd) : (i += 4) {
            simd.store4_f32(dst, i, simd.load4_f32(src, i + 1) - simd.load4_f32(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for i64.
export fn diff_2d_i64(a: [*]const i64, out: [*]i64, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        var i: u32 = 0;
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for i32.
export fn diff_2d_i32(a: [*]const i32, out: [*]i32, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 3);
        var i: u32 = 0;
        while (i < n_simd) : (i += 4) {
            simd.store4_i32(dst, i, simd.load4_i32(src, i + 1) - simd.load4_i32(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for i16.
export fn diff_2d_i16(a: [*]const i16, out: [*]i16, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 7);
        var i: u32 = 0;
        while (i < n_simd) : (i += 8) {
            simd.store8_i16(dst, i, simd.load8_i16(src, i + 1) - simd.load8_i16(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for i8.
export fn diff_2d_i8(a: [*]const i8, out: [*]i8, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 15);
        var i: u32 = 0;
        while (i < n_simd) : (i += 16) {
            simd.store16_i8(dst, i, simd.load16_i8(src, i + 1) - simd.load16_i8(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

// --- Tests ---

test "diff_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 3.0, 6.0, 10.0, 15.0 };
    var out: [4]f64 = undefined;
    diff_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 5.0, 1e-10);
}

test "diff_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, 4, 9, 16 };
    var out: [4]i32 = undefined;
    diff_i32(&a, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[2], 5);
    try testing.expectEqual(out[3], 7);
}

test "diff_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 121, 123 };
    var out: [17]i8 = undefined;
    diff_i8(&a, &out, 17);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 3);
    try testing.expectEqual(out[16], 2);
}

test "diff_2d_f64 basic" {
    const testing = @import("std").testing;
    // 2x4 → 2x3
    const a = [_]f64{ 1.0, 3.0, 6.0, 10.0, 0.0, 2.0, 5.0, 9.0 };
    var out: [6]f64 = undefined;
    diff_2d_f64(&a, &out, 2, 4);
    // row 0: [2, 3, 4]
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-10);
    // row 1: [2, 3, 4]
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 4.0, 1e-10);
}

test "diff_2d_i32 basic" {
    const testing = @import("std").testing;
    // 2x3 → 2x2
    const a = [_]i32{ 1, 4, 9, 10, 20, 30 };
    var out: [4]i32 = undefined;
    diff_2d_i32(&a, &out, 2, 3);
    try testing.expectEqual(out[0], 3);
    try testing.expectEqual(out[1], 5);
    try testing.expectEqual(out[2], 10);
    try testing.expectEqual(out[3], 10);
}

test "diff_f64 SIMD boundary N=1" {
    const testing = @import("std").testing;
    const a = [_]f64{ 5.0, 8.0 };
    var out: [1]f64 = undefined;
    diff_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-10);
}

test "diff_f32 SIMD boundary N=7" {
    const testing = @import("std").testing;
    var a: [8]f32 = undefined;
    for (0..8) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(i));
    }
    var out: [7]f32 = undefined;
    diff_f32(&a, &out, 7);
    // diff of squares: 1, 3, 5, 7, 9, 11, 13
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[6], 13.0, 1e-5);
}

test "diff_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 10, 30, 60, 100 };
    var out: [3]i64 = undefined;
    diff_i64(&a, &out, 3);
    try testing.expectEqual(out[0], 20);
    try testing.expectEqual(out[1], 30);
    try testing.expectEqual(out[2], 40);
}

test "diff_i16 SIMD boundary N=9" {
    const testing = @import("std").testing;
    var a: [10]i16 = undefined;
    for (0..10) |i| {
        a[i] = @intCast(i * i);
    }
    var out: [9]i16 = undefined;
    diff_i16(&a, &out, 9);
    // diff of squares: 1, 3, 5, 7, 9, 11, 13, 15, 17
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[8], 17);
}

test "diff_2d_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 3, 6, 10, 0, 2, 5, 9 };
    var out: [6]f32 = undefined;
    diff_2d_f32(&a, &out, 2, 4);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-5);
}

test "diff_2d_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 4, 9, 10, 20, 30 };
    var out: [4]i16 = undefined;
    diff_2d_i16(&a, &out, 2, 3);
    try testing.expectEqual(out[0], 3);
    try testing.expectEqual(out[1], 5);
    try testing.expectEqual(out[2], 10);
    try testing.expectEqual(out[3], 10);
}

test "diff_2d_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 3, 6, 2, 5, 9 };
    var out: [4]i8 = undefined;
    diff_2d_i8(&a, &out, 2, 3);
    try testing.expectEqual(out[0], 2);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[2], 3);
    try testing.expectEqual(out[3], 4);
}

test "diff_2d_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 10, 30, 60, 5, 15, 45 };
    var out: [4]i64 = undefined;
    diff_2d_i64(&a, &out, 2, 3);
    try testing.expectEqual(out[0], 20);
    try testing.expectEqual(out[1], 30);
    try testing.expectEqual(out[2], 10);
    try testing.expectEqual(out[3], 30);
}

test "diff_f64 SIMD boundary N=3 (V2f64 remainder)" {
    const testing = @import("std").testing;
    // N=3 output: 2 processed by SIMD (V2f64), 1 by scalar remainder
    const a = [_]f64{ 1.0, 4.0, 9.0, 16.0 };
    var out: [3]f64 = undefined;
    diff_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 7.0, 1e-10);
}

test "diff_f32 SIMD boundary N=7 (V4f32 remainder)" {
    const testing = @import("std").testing;
    // N=7: 4 by SIMD, 3 by scalar remainder
    const a = [_]f32{ 2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0, 30.0 };
    var out: [7]f32 = undefined;
    diff_f32(&a, &out, 7);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[5], 6.0, 1e-5);
    try testing.expectApproxEqAbs(out[6], 7.0, 1e-5);
}

test "diff_i16 SIMD boundary N=9 (V8i16 remainder)" {
    const testing = @import("std").testing;
    // N=9: 8 by SIMD, 1 by scalar remainder
    const a = [_]i16{ 10, 20, 33, 49, 68, 90, 115, 143, 174, 208 };
    var out: [9]i16 = undefined;
    diff_i16(&a, &out, 9);
    try testing.expectEqual(out[0], 10);
    try testing.expectEqual(out[1], 13);
    try testing.expectEqual(out[2], 16);
    try testing.expectEqual(out[3], 19);
    try testing.expectEqual(out[4], 22);
    try testing.expectEqual(out[5], 25);
    try testing.expectEqual(out[6], 28);
    try testing.expectEqual(out[7], 31);
    try testing.expectEqual(out[8], 34);
}

test "diff_i64 scalar path" {
    const testing = @import("std").testing;
    // i64 uses pure scalar loop (no SIMD)
    const a = [_]i64{ 100, 250, 480, 800, 1200 };
    var out: [4]i64 = undefined;
    diff_i64(&a, &out, 4);
    try testing.expectEqual(out[0], 150);
    try testing.expectEqual(out[1], 230);
    try testing.expectEqual(out[2], 320);
    try testing.expectEqual(out[3], 400);
}

test "diff_2d_f32 multi-row" {
    const testing = @import("std").testing;
    // 3x3 → 3x2
    const a = [_]f32{ 1.0, 4.0, 9.0, 2.0, 6.0, 12.0, 3.0, 8.0, 15.0 };
    var out: [6]f32 = undefined;
    diff_2d_f32(&a, &out, 3, 3);
    // row 0: [3, 5]
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 5.0, 1e-5);
    // row 1: [4, 6]
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 6.0, 1e-5);
    // row 2: [5, 7]
    try testing.expectApproxEqAbs(out[4], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[5], 7.0, 1e-5);
}

test "diff_2d_i8 multi-row" {
    const testing = @import("std").testing;
    // 2x4 → 2x3
    const a = [_]i8{ 1, 3, 7, 15, 2, 5, 11, 20 };
    var out: [6]i8 = undefined;
    diff_2d_i8(&a, &out, 2, 4);
    // row 0: [2, 4, 8]
    try testing.expectEqual(out[0], 2);
    try testing.expectEqual(out[1], 4);
    try testing.expectEqual(out[2], 8);
    // row 1: [3, 6, 9]
    try testing.expectEqual(out[3], 3);
    try testing.expectEqual(out[4], 6);
    try testing.expectEqual(out[5], 9);
}

test "diff_2d_i16 multi-row" {
    const testing = @import("std").testing;
    // 2x4 → 2x3
    const a = [_]i16{ 10, 30, 60, 100, 5, 15, 35, 70 };
    var out: [6]i16 = undefined;
    diff_2d_i16(&a, &out, 2, 4);
    // row 0: [20, 30, 40]
    try testing.expectEqual(out[0], 20);
    try testing.expectEqual(out[1], 30);
    try testing.expectEqual(out[2], 40);
    // row 1: [10, 20, 35]
    try testing.expectEqual(out[3], 10);
    try testing.expectEqual(out[4], 20);
    try testing.expectEqual(out[5], 35);
}

test "diff single element (N=1 output, 2-element input)" {
    const testing = @import("std").testing;
    // f64
    const a_f64 = [_]f64{ 7.5, 12.5 };
    var out_f64: [1]f64 = undefined;
    diff_f64(&a_f64, &out_f64, 1);
    try testing.expectApproxEqAbs(out_f64[0], 5.0, 1e-10);

    // f32
    const a_f32 = [_]f32{ 7.5, 12.5 };
    var out_f32: [1]f32 = undefined;
    diff_f32(&a_f32, &out_f32, 1);
    try testing.expectApproxEqAbs(out_f32[0], 5.0, 1e-5);

    // i32
    const a_i32 = [_]i32{ 7, 12 };
    var out_i32: [1]i32 = undefined;
    diff_i32(&a_i32, &out_i32, 1);
    try testing.expectEqual(out_i32[0], 5);

    // i64
    const a_i64 = [_]i64{ 7, 12 };
    var out_i64: [1]i64 = undefined;
    diff_i64(&a_i64, &out_i64, 1);
    try testing.expectEqual(out_i64[0], 5);
}

test "diff_f64 edge Inf differences" {
    const testing = @import("std").testing;
    const inf = @as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    const neg_inf = -inf;

    // diff([0, Inf]) = [Inf]
    const a1 = [_]f64{ 0.0, inf };
    var out1: [1]f64 = undefined;
    diff_f64(&a1, &out1, 1);
    try testing.expect(out1[0] == inf);

    // diff([Inf, 0]) = [-Inf]
    const a2 = [_]f64{ inf, 0.0 };
    var out2: [1]f64 = undefined;
    diff_f64(&a2, &out2, 1);
    try testing.expect(out2[0] == neg_inf);

    // diff([Inf, Inf]) = NaN (Inf - Inf)
    const a3 = [_]f64{ inf, inf };
    var out3: [1]f64 = undefined;
    diff_f64(&a3, &out3, 1);
    try testing.expect(out3[0] != out3[0]); // NaN != NaN
}
