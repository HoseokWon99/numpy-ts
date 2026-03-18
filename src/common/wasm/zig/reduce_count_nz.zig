//! WASM reduction count-nonzero kernels for all numeric types.
//!
//! Reduction: result = count of a[i] != 0 for i in 0..N
//! No unsigned variants needed — non-zero check is sign-agnostic.

const simd = @import("simd.zig");

/// Returns the count of non-zero f64 elements.
/// Note: NaN is considered non-zero, so reduce_count_nz_f64([NaN], 1) returns 1.
export fn reduce_count_nz_f64(a: [*]const f64, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero f32 elements.
/// Note: NaN is considered non-zero, so reduce_count_nz_f32([NaN], 1) returns 1.
export fn reduce_count_nz_f32(a: [*]const f32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero i64 elements.
/// Handles both signed (i64) and unsigned (u64).
export fn reduce_count_nz_i64(a: [*]const i64, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero i32 elements.
/// Handles both signed (i32) and unsigned (u32).
export fn reduce_count_nz_i32(a: [*]const i32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero i16 elements.
/// Handles both signed (i16) and unsigned (u16).
export fn reduce_count_nz_i16(a: [*]const i16, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero i8 elements.
/// Handles both signed (i8) and unsigned (u8).
export fn reduce_count_nz_i8(a: [*]const i8, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) count += 1;
    }
    return count;
}

// --- Tests ---

test "reduce_count_nz_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 0.0, 3.0, 0.0, 5.0 };
    try testing.expectEqual(reduce_count_nz_f64(&a, 5), 3);
}

test "reduce_count_nz_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 0, 3, 0, 5 };
    try testing.expectEqual(reduce_count_nz_i32(&a, 5), 2);
}

test "reduce_count_nz_f64 all zero" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 0.0, 0.0 };
    try testing.expectEqual(reduce_count_nz_f64(&a, 3), 0);
}

test "reduce_count_nz_f64 all nonzero" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    try testing.expectEqual(reduce_count_nz_f64(&a, 3), 3);
}

test "reduce_count_nz_f64 empty" {
    const testing = @import("std").testing;
    const a = [_]f64{};
    try testing.expectEqual(reduce_count_nz_f64(&a, 0), 0);
}

test "reduce_count_nz_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, 0.0, 2.0 };
    try testing.expectEqual(reduce_count_nz_f32(&a, 4), 2);
}

test "reduce_count_nz_i64 negatives are nonzero" {
    const testing = @import("std").testing;
    const a = [_]i64{ -1, 0, -2, 0 };
    try testing.expectEqual(reduce_count_nz_i64(&a, 4), 2);
}

test "reduce_count_nz_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 100, 0, 200, 0 };
    try testing.expectEqual(reduce_count_nz_i16(&a, 5), 2);
}

test "reduce_count_nz_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 0, -1, 0 };
    try testing.expectEqual(reduce_count_nz_i8(&a, 4), 2);
}
