//! WASM reduction argmax kernels for all numeric types.
//!
//! Reduction: result = index of max(a[0..N])
//! Returns u32 index. Unsigned variants needed — comparison is sign-dependent.
//! Scalar loops only (tracking index alongside value makes SIMD complex).

/// Returns the index of the maximum f64 element. Returns 0 if N=0.
/// Note: NaN is considered less than any number, so reduce_argmax_f64([NaN], 1) returns 0.
export fn reduce_argmax_f64(a: [*]const f64, N: u32) u32 {
    if (N == 0) return 0;
    var best: f64 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum f32 element. Returns 0 if N=0.
/// Note: NaN is considered less than any number, so reduce_argmax_f32([NaN], 1) returns 0.
export fn reduce_argmax_f32(a: [*]const f32, N: u32) u32 {
    if (N == 0) return 0;
    var best: f32 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum i64 element. Returns 0 if N=0.
export fn reduce_argmax_i64(a: [*]const i64, N: u32) u32 {
    if (N == 0) return 0;
    var best: i64 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum u64 element. Returns 0 if N=0.
export fn reduce_argmax_u64(a: [*]const u64, N: u32) u32 {
    if (N == 0) return 0;
    var best: u64 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum i32 element. Returns 0 if N=0.
export fn reduce_argmax_i32(a: [*]const i32, N: u32) u32 {
    if (N == 0) return 0;
    var best: i32 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum u32 element. Returns 0 if N=0.
export fn reduce_argmax_u32(a: [*]const u32, N: u32) u32 {
    if (N == 0) return 0;
    var best: u32 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum i16 element. Returns 0 if N=0.
export fn reduce_argmax_i16(a: [*]const i16, N: u32) u32 {
    if (N == 0) return 0;
    var best: i16 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum u16 element. Returns 0 if N=0.
export fn reduce_argmax_u16(a: [*]const u16, N: u32) u32 {
    if (N == 0) return 0;
    var best: u16 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum i8 element. Returns 0 if N=0.
export fn reduce_argmax_i8(a: [*]const i8, N: u32) u32 {
    if (N == 0) return 0;
    var best: i8 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum u8 element. Returns 0 if N=0.
export fn reduce_argmax_u8(a: [*]const u8, N: u32) u32 {
    if (N == 0) return 0;
    var best: u8 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

// --- Tests ---

test "reduce_argmax_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0 };
    try testing.expectEqual(reduce_argmax_f64(&a, 5), 4);
}

test "reduce_argmax_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 3, 200, 4, 1, 5 };
    try testing.expectEqual(reduce_argmax_u8(&a, 5), 1);
}

test "reduce_argmax_f64 max at first and single" {
    const testing = @import("std").testing;
    const a = [_]f64{ 10.0, 1.0, 2.0 };
    try testing.expectEqual(reduce_argmax_f64(&a, 3), 0);
    const b = [_]f64{42.0};
    try testing.expectEqual(reduce_argmax_f64(&b, 1), 0);
}

test "reduce_argmax_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 5.0, 3.0 };
    try testing.expectEqual(reduce_argmax_f32(&a, 3), 1);
}

test "reduce_argmax_i64 negatives" {
    const testing = @import("std").testing;
    const a = [_]i64{ -5, -1, -3 };
    try testing.expectEqual(reduce_argmax_i64(&a, 3), 1);
}

test "reduce_argmax_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 7, 2, 9, 4 };
    try testing.expectEqual(reduce_argmax_i32(&a, 4), 2);
}

test "reduce_argmax_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ -10, 100, 50 };
    try testing.expectEqual(reduce_argmax_i16(&a, 3), 1);
}

test "reduce_argmax_i8 negatives" {
    const testing = @import("std").testing;
    const a = [_]i8{ -128, -1, -64 };
    try testing.expectEqual(reduce_argmax_i8(&a, 3), 1);
}

test "reduce_argmax_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 5, 100, 3 };
    try testing.expectEqual(reduce_argmax_u64(&a, 3), 1);
}

test "reduce_argmax_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 1, 2, 0xFFFFFFFF };
    try testing.expectEqual(reduce_argmax_u32(&a, 3), 2);
}

test "reduce_argmax_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 0xFFFF, 0, 1 };
    try testing.expectEqual(reduce_argmax_u16(&a, 3), 0);
}

test "reduce_argmax ties return first occurrence" {
    const testing = @import("std").testing;
    const a = [_]i32{ 5, 5, 5 };
    try testing.expectEqual(reduce_argmax_i32(&a, 3), 0);
}
