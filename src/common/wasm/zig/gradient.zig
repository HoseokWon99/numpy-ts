//! WASM 1D gradient kernel using central differences.
//!
//! out[0]     = (a[1] - a[0]) / h          (forward)
//! out[i]     = (a[i+1] - a[i-1]) / (2*h)  (central)
//! out[N-1]   = (a[N-1] - a[N-2]) / h      (backward)
//!
//! Output is always f64 (gradient promotes to float).

const simd = @import("simd.zig");

/// 1D gradient for f64 input, f64 output.
export fn gradient_f64(a: [*]const f64, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    // Forward difference at start
    out[0] = (a[1] - a[0]) / h;

    // Central differences in interior (SIMD)
    const h2 = 2.0 * h;
    const h2v: simd.V2f64 = @splat(h2);
    const interior = N - 1;
    const n_simd = 1 + ((interior - 1) & ~@as(u32, 1)); // round down to even, offset by 1
    var i: u32 = 1;
    while (i < n_simd and i < interior) : (i += 2) {
        const vp = simd.load2_f64(a, i + 1);
        const vm = simd.load2_f64(a, i - 1);
        simd.store2_f64(out, i, (vp - vm) / h2v);
    }
    while (i < interior) : (i += 1) {
        out[i] = (a[i + 1] - a[i - 1]) / h2;
    }

    // Backward difference at end
    out[N - 1] = (a[N - 1] - a[N - 2]) / h;
}

/// 1D gradient for f32 input, f32 output.
export fn gradient_f32(a: [*]const f32, out: [*]f32, N: u32, h: f32) void {
    if (N < 2) return;

    out[0] = (a[1] - a[0]) / h;

    const h2 = 2.0 * h;
    const h2v: simd.V4f32 = @splat(h2);
    const interior = N - 1;
    const n_simd = 1 + ((interior - 1) & ~@as(u32, 3));
    var i: u32 = 1;
    while (i < n_simd and i < interior) : (i += 4) {
        const vp = simd.load4_f32(a, i + 1);
        const vm = simd.load4_f32(a, i - 1);
        simd.store4_f32(out, i, (vp - vm) / h2v);
    }
    while (i < interior) : (i += 1) {
        out[i] = (a[i + 1] - a[i - 1]) / h2;
    }

    out[N - 1] = (a[N - 1] - a[N - 2]) / h;
}

/// 1D gradient for i64 input → f64 output (scalar loop).
export fn gradient_i64(a: [*]const i64, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;

    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }

    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for i32 input → f64 output.
export fn gradient_i32(a: [*]const i32, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;

    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }

    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for i16 input → f64 output.
export fn gradient_i16(a: [*]const i16, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;

    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }

    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for i8 input → f64 output.
export fn gradient_i8(a: [*]const i8, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;

    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }

    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

// --- Tests ---

test "gradient_f64 basic" {
    const testing = @import("std").testing;
    // a = [0, 1, 4, 9, 16] (x^2 at 0..4)
    const a = [_]f64{ 0.0, 1.0, 4.0, 9.0, 16.0 };
    var out: [5]f64 = undefined;
    gradient_f64(&a, &out, 5, 1.0);
    // forward:  (1-0)/1 = 1
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    // central:  (4-0)/2 = 2
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    // central:  (9-1)/2 = 4
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-10);
    // central:  (16-4)/2 = 6
    try testing.expectApproxEqAbs(out[3], 6.0, 1e-10);
    // backward: (16-9)/1 = 7
    try testing.expectApproxEqAbs(out[4], 7.0, 1e-10);
}

test "gradient_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, 4.0, 9.0, 16.0 };
    var out: [5]f32 = undefined;
    gradient_f32(&a, &out, 5, 1.0);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 7.0, 1e-5);
}

test "gradient_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, 4, 9, 16 };
    var out: [5]f64 = undefined;
    gradient_i32(&a, &out, 5, 1.0);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 7.0, 1e-10);
}
