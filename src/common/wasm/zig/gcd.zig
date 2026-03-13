//! WASM element-wise GCD (greatest common divisor) kernels.
//!
//! Scalar: out[i] = gcd(a[i], scalar)
//! Binary: out[i] = gcd(a[i], b[i])
//! Uses Euclidean algorithm. Operates on contiguous 1D buffers.

/// Scalar GCD for i32: out[i] = gcd(abs(a[i]), abs(scalar)).
export fn gcd_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    const b = if (scalar < 0) -scalar else scalar;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var x = a[i];
        if (x < 0) x = -x;
        var y = b;
        while (y != 0) {
            const temp = y;
            y = @rem(x, y);
            x = temp;
        }
        out[i] = x;
    }
}

/// Binary GCD for i32: out[i] = gcd(abs(a[i]), abs(b[i])).
export fn gcd_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var x = a[i];
        if (x < 0) x = -x;
        var y = b[i];
        if (y < 0) y = -y;
        while (y != 0) {
            const temp = y;
            y = @rem(x, y);
            x = temp;
        }
        out[i] = x;
    }
}

// --- Tests ---

test "gcd_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 18, 7, 0, -15 };
    var out: [5]i32 = undefined;
    gcd_scalar_i32(&a, &out, 5, 6);
    try testing.expectEqual(out[0], 6); // gcd(12,6)
    try testing.expectEqual(out[1], 6); // gcd(18,6)
    try testing.expectEqual(out[2], 1); // gcd(7,6)
    try testing.expectEqual(out[3], 6); // gcd(0,6)
    try testing.expectEqual(out[4], 3); // gcd(15,6)
}

test "gcd_i32 binary" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 18, 7, 0 };
    const b = [_]i32{ 8, 12, 5, 3 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 4); // gcd(12,8)
    try testing.expectEqual(out[1], 6); // gcd(18,12)
    try testing.expectEqual(out[2], 1); // gcd(7,5)
    try testing.expectEqual(out[3], 3); // gcd(0,3)
}

test "gcd_i32 edge case gcd(0,0)" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    const b = [_]i32{0};
    var out: [1]i32 = undefined;
    gcd_i32(&a, &b, &out, 1);
    try testing.expectEqual(out[0], 0); // gcd(0,0) = 0
}

test "gcd_i32 edge case gcd(x,0) and gcd(0,x)" {
    const testing = @import("std").testing;
    const a = [_]i32{ 15, 0, 0, 42 };
    const b = [_]i32{ 0, 15, 0, 0 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 15); // gcd(15,0)
    try testing.expectEqual(out[1], 15); // gcd(0,15)
    try testing.expectEqual(out[2], 0); // gcd(0,0)
    try testing.expectEqual(out[3], 42); // gcd(42,0)
}

test "gcd_i32 gcd(1,x) = 1" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 1, 1, 1 };
    const b = [_]i32{ 7, 100, 999, 1 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "gcd_i32 large coprime numbers" {
    const testing = @import("std").testing;
    const a = [_]i32{ 97, 1000003 };
    const b = [_]i32{ 89, 999979 };
    var out: [2]i32 = undefined;
    gcd_i32(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 1); // 97 and 89 are both prime
    try testing.expectEqual(out[1], 1); // coprime large numbers
}

test "gcd_i32 identity gcd(x,x) = x" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 7, 42, 100 };
    const b = [_]i32{ 1, 7, 42, 100 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 7);
    try testing.expectEqual(out[2], 42);
    try testing.expectEqual(out[3], 100);
}

test "gcd_i32 negative inputs" {
    const testing = @import("std").testing;
    const a = [_]i32{ -12, 12, -12, -18 };
    const b = [_]i32{ 8, -8, -8, -12 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 4); // gcd(|-12|,|8|)
    try testing.expectEqual(out[1], 4); // gcd(|12|,|-8|)
    try testing.expectEqual(out[2], 4); // gcd(|-12|,|-8|)
    try testing.expectEqual(out[3], 6); // gcd(|-18|,|-12|)
}

test "gcd_scalar_i32 gcd(x,0) returns abs(x)" {
    const testing = @import("std").testing;
    const a = [_]i32{ 5, -7, 0, 13 };
    var out: [4]i32 = undefined;
    gcd_scalar_i32(&a, &out, 4, 0);
    try testing.expectEqual(out[0], 5);
    try testing.expectEqual(out[1], 7);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 13);
}

test "gcd_scalar_i32 gcd(x,1) = 1" {
    const testing = @import("std").testing;
    const a = [_]i32{ 5, 100, -7, 0 };
    var out: [4]i32 = undefined;
    gcd_scalar_i32(&a, &out, 4, 1);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "gcd_scalar_i32 negative scalar" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 18, -15 };
    var out: [3]i32 = undefined;
    gcd_scalar_i32(&a, &out, 3, -6);
    try testing.expectEqual(out[0], 6); // gcd(12,6)
    try testing.expectEqual(out[1], 6); // gcd(18,6)
    try testing.expectEqual(out[2], 3); // gcd(15,6)
}

test "gcd_i32 various known values" {
    const testing = @import("std").testing;
    const a = [_]i32{ 48, 54, 35, 100, 17, 144, 1000 };
    const b = [_]i32{ 18, 24, 15, 75, 13, 89, 600 };
    var out: [7]i32 = undefined;
    gcd_i32(&a, &b, &out, 7);
    try testing.expectEqual(out[0], 6); // gcd(48,18)
    try testing.expectEqual(out[1], 6); // gcd(54,24)
    try testing.expectEqual(out[2], 5); // gcd(35,15)
    try testing.expectEqual(out[3], 25); // gcd(100,75)
    try testing.expectEqual(out[4], 1); // gcd(17,13) both prime
    try testing.expectEqual(out[5], 1); // gcd(144,89) 89 is prime
    try testing.expectEqual(out[6], 200); // gcd(1000,600)
}
