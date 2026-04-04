//! WASM reduction quantile kernel.
//!
//! Computes the q-th quantile of a contiguous f64 array.
//! The TS wrapper converts all dtypes to f64 before calling this kernel.
//! Uses in-place sort then linear interpolation (matching NumPy's default method).

const std = @import("std");

/// Compute the q-th quantile of a contiguous f64 array.
/// The input buffer `a` is modified in-place (sorted).
/// Returns the interpolated quantile value.
export fn reduce_quantile_f64(a: [*]f64, N: u32, q: f64) f64 {
    if (N == 0) return 0;
    if (N == 1) return a[0];

    // Sort in-place using Zig's pdqsort (pattern-defeating quicksort)
    const slice = a[0..@as(usize, N)];
    std.mem.sortUnstable(f64, slice, {}, std.sort.asc(f64));

    // Linear interpolation (NumPy default method='linear')
    const idx = q * @as(f64, @floatFromInt(N - 1));
    const lower = @as(usize, @intFromFloat(@floor(idx)));
    const upper = @as(usize, @intFromFloat(@ceil(idx)));

    if (lower == upper) return a[lower];

    const frac = idx - @as(f64, @floatFromInt(lower));
    return a[lower] * (1.0 - frac) + a[upper] * frac;
}

/// Strided quantile: for each output position (outer × inner), sort the axis slice and interpolate.
/// Input a is [outer × axis × inner] contiguous f64. Output is [outer × inner] f64.
/// Uses a scratch buffer for sorting each column (does NOT modify input).
export fn reduce_quantile_strided_f64(a: [*]const f64, out: [*]f64, scratch: [*]f64, outer: u32, axis: u32, inner: u32, q: f64) void {
    const stride = @as(usize, axis) * @as(usize, inner);
    var o: u32 = 0;
    while (o < outer) : (o += 1) {
        var inn: u32 = 0;
        while (inn < inner) : (inn += 1) {
            // Gather column into scratch
            const base = @as(usize, o) * stride + @as(usize, inn);
            var k: u32 = 0;
            while (k < axis) : (k += 1) {
                scratch[k] = a[base + @as(usize, k) * @as(usize, inner)];
            }
            // Sort scratch
            std.mem.sortUnstable(f64, scratch[0..axis], {}, std.sort.asc(f64));
            // Interpolate
            const idx = q * @as(f64, @floatFromInt(axis - 1));
            const lower = @as(usize, @intFromFloat(@floor(idx)));
            const upper = @as(usize, @intFromFloat(@ceil(idx)));
            if (lower == upper) {
                out[@as(usize, o) * @as(usize, inner) + @as(usize, inn)] = scratch[lower];
            } else {
                const frac = idx - @as(f64, @floatFromInt(lower));
                out[@as(usize, o) * @as(usize, inner) + @as(usize, inn)] = scratch[lower] * (1.0 - frac) + scratch[upper] * frac;
            }
        }
    }
}

// --- Tests ---

test "reduce_quantile_f64 median" {
    const testing = std.testing;
    var a = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 5, 0.5), 3.0, 1e-10);
}

test "reduce_quantile_f64 q=0.25" {
    const testing = std.testing;
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    // q=0.25: idx=1.0 → exactly a[1]=2.0
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 5, 0.25), 2.0, 1e-10);
}

test "reduce_quantile_f64 interpolation" {
    const testing = std.testing;
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    // q=0.5: idx=1.5 → interpolate between 2.0 and 3.0 → 2.5
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 4, 0.5), 2.5, 1e-10);
}

test "reduce_quantile_f64 q=0 is min q=1 is max" {
    const testing = std.testing;
    var a = [_]f64{ 3.0, 1.0, 5.0, 2.0, 4.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 5, 0.0), 1.0, 1e-10);
    var b = [_]f64{ 3.0, 1.0, 5.0, 2.0, 4.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&b, 5, 1.0), 5.0, 1e-10);
}

test "reduce_quantile_f64 single element" {
    const testing = std.testing;
    var a = [_]f64{42.0};
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 1, 0.5), 42.0, 1e-10);
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 1, 0.0), 42.0, 1e-10);
}

test "reduce_quantile_f64 empty returns zero" {
    const testing = std.testing;
    var a = [_]f64{};
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 0, 0.5), 0.0, 1e-10);
}

test "reduce_quantile_f64 constant array" {
    const testing = std.testing;
    var a = [_]f64{ 7.0, 7.0, 7.0, 7.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 4, 0.25), 7.0, 1e-10);
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 4, 0.75), 7.0, 1e-10);
}

test "reduce_quantile_f64 already sorted" {
    const testing = std.testing;
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 5, 0.5), 3.0, 1e-10);
}

test "reduce_quantile_f64 negatives" {
    const testing = std.testing;
    var a = [_]f64{ -5.0, -3.0, -1.0 };
    // median of [-5,-3,-1] = -3
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 3, 0.5), -3.0, 1e-10);
}

test "reduce_quantile_f64 two elements interpolation" {
    const testing = std.testing;
    // q=0.5: idx=0.5, interpolate between a[0]=1 and a[1]=3 → 2
    var a = [_]f64{ 3.0, 1.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 2, 0.5), 2.0, 1e-10);
}

test "reduce_quantile_strided_f64 basic 2x3x2" {
    const testing = std.testing;
    // Shape [1, 3, 2] — outer=1, axis=3, inner=2
    // Column 0: a[0], a[2], a[4] = 6, 2, 4 → sorted [2,4,6] → median = 4
    // Column 1: a[1], a[3], a[5] = 1, 5, 3 → sorted [1,3,5] → median = 3
    var a = [_]f64{ 6.0, 1.0, 2.0, 5.0, 4.0, 3.0 };
    var out: [2]f64 = undefined;
    var scratch: [3]f64 = undefined;
    reduce_quantile_strided_f64(&a, &out, &scratch, 1, 3, 2, 0.5);
    try testing.expectApproxEqAbs(out[0], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
}

test "reduce_quantile_strided_f64 outer=2" {
    const testing = std.testing;
    // Shape [2, 3, 1] — outer=2, axis=3, inner=1
    // Slice 0: a[0..3] = [5, 1, 3] → sorted [1,3,5] → median = 3
    // Slice 1: a[3..6] = [6, 2, 4] → sorted [2,4,6] → median = 4
    var a = [_]f64{ 5.0, 1.0, 3.0, 6.0, 2.0, 4.0 };
    var out: [2]f64 = undefined;
    var scratch: [3]f64 = undefined;
    reduce_quantile_strided_f64(&a, &out, &scratch, 2, 3, 1, 0.5);
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
}

test "reduce_quantile_strided_f64 interpolation q=0.25" {
    const testing = std.testing;
    // Shape [1, 4, 1] — outer=1, axis=4, inner=1
    // Values: [1, 2, 3, 4] → q=0.25: idx=0.75 → interp(1, 2, 0.75) = 1.75
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var out: [1]f64 = undefined;
    var scratch: [4]f64 = undefined;
    reduce_quantile_strided_f64(&a, &out, &scratch, 1, 4, 1, 0.25);
    try testing.expectApproxEqAbs(out[0], 1.75, 1e-10);
}
