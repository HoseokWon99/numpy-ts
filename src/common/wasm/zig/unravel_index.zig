/// Unravel flat indices into multi-dimensional indices (i32/u32).
/// indices: flat input indices, out: output (ndim * N i32 values, row-major by dimension),
/// strides: precomputed strides (ndim i32 values), shape: dimension sizes (ndim i32 values).
/// Works for both i32 and u32 indices (u32 reinterprets as i32 — safe since indices are non-negative).
export fn unravel_index_i32(indices: [*]const i32, out: [*]i32, N: u32, strides: [*]const i32, shape: [*]const i32, ndim: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var remaining: i32 = indices[i];
        var d: u32 = 0;
        while (d < ndim) : (d += 1) {
            const s = strides[d];
            const idx = @divTrunc(remaining, s);
            out[d * N + i] = @rem(idx, shape[d]);
            remaining = @rem(remaining, s);
        }
    }
}

/// Unravel flat indices into multi-dimensional indices (i64/u64).
/// Output is i64 to match NumPy's int64 result dtype.
/// Works for both i64 and u64 indices (indices are always non-negative).
export fn unravel_index_i64(indices: [*]const i64, out: [*]i64, N: u32, strides: [*]const i64, shape: [*]const i64, ndim: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var remaining: i64 = indices[i];
        var d: u32 = 0;
        while (d < ndim) : (d += 1) {
            const s = strides[d];
            const idx = @divTrunc(remaining, s);
            out[d * N + i] = @rem(idx, shape[d]);
            remaining = @rem(remaining, s);
        }
    }
}

// --- Tests ---

test "unravel_index_i32 basic 2D" {
    const testing = @import("std").testing;
    // shape [3, 4], strides [4, 1] (C-order)
    const indices = [_]i32{ 0, 5, 11 };
    const strides = [_]i32{ 4, 1 };
    const shape = [_]i32{ 3, 4 };
    var out: [6]i32 = undefined; // 2 dims * 3 indices
    unravel_index_i32(&indices, &out, 3, &strides, &shape, 2);
    // Index 0 -> (0, 0)
    try testing.expectEqual(out[0], 0); // dim0[0]
    try testing.expectEqual(out[3], 0); // dim1[0]
    // Index 5 -> (1, 1)
    try testing.expectEqual(out[1], 1); // dim0[1]
    try testing.expectEqual(out[4], 1); // dim1[1]
    // Index 11 -> (2, 3)
    try testing.expectEqual(out[2], 2); // dim0[2]
    try testing.expectEqual(out[5], 3); // dim1[2]
}

test "unravel_index_i32 3D" {
    const testing = @import("std").testing;
    // shape [2, 3, 4], strides [12, 4, 1]
    const indices = [_]i32{ 0, 23 };
    const strides = [_]i32{ 12, 4, 1 };
    const shape = [_]i32{ 2, 3, 4 };
    var out: [6]i32 = undefined; // 3 dims * 2 indices
    unravel_index_i32(&indices, &out, 2, &strides, &shape, 3);
    // Index 0 -> (0, 0, 0)
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[4], 0);
    // Index 23 -> (1, 2, 3)
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[3], 2);
    try testing.expectEqual(out[5], 3);
}
