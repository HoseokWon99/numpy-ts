//! WASM conjugate dot product kernels for complex types.
//!
//! Computes out = conj(a) · b = sum_k conj(a[k]) * b[k].
//! For complex128: each element = 2 f64s (re, im).
//! For complex64: each element = 2 f32s (re, im).
//! Real-type vdot is identical to dot — use dot kernels instead.

/// Computes the conjugate dot product of two complex128 vectors of length K.
/// K is the number of complex elements (each = 2 f64s).
/// conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
///             = (a_re*b_re + a_im*b_im) + (a_re*b_im - a_im*b_re)*i
export fn vdot_c128(a: [*]const f64, b: [*]const f64, out: [*]f64, K: u32) void {
    var sum_re: f64 = 0;
    var sum_im: f64 = 0;

    // Scalar loop: conjugate complex multiply-accumulate
    for (0..K) |k| {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        // conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
        sum_re += a_re * b_re + a_im * b_im;
        sum_im += a_re * b_im - a_im * b_re;
    }
    out[0] = sum_re;
    out[1] = sum_im;
}

/// Computes the conjugate dot product of two complex64 vectors of length K.
/// K is the number of complex elements (each = 2 f32s).
/// conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
///             = (a_re*b_re + a_im*b_im) + (a_re*b_im - a_im*b_re)*i
export fn vdot_c64(a: [*]const f32, b: [*]const f32, out: [*]f32, K: u32) void {
    var sum_re: f32 = 0;
    var sum_im: f32 = 0;

    // Scalar loop: conjugate complex multiply-accumulate
    for (0..K) |k| {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        // conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
        sum_re += a_re * b_re + a_im * b_im;
        sum_im += a_re * b_im - a_im * b_re;
    }
    out[0] = sum_re;
    out[1] = sum_im;
}

// --- Tests ---

test "vdot_c128 conjugate" {
    const testing = @import("std").testing;
    // conj(1+2i) * (3+4i) = (1-2i)*(3+4i) = 3+4i-6i+8 = 11-2i
    const a = [_]f64{ 1, 2 };
    const b = [_]f64{ 3, 4 };
    var out: [2]f64 = undefined;
    vdot_c128(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 11.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-10);
}

test "vdot_c128 two elements" {
    const testing = @import("std").testing;
    // conj(1+2i)*(3+4i) + conj(5+6i)*(7+8i)
    // = (11-2i) + (5-6i)*(7+8i) = (11-2i) + (35+40i-42i+48) = (11-2i)+(83-2i) = 94-4i
    const a = [_]f64{ 1, 2, 5, 6 };
    const b = [_]f64{ 3, 4, 7, 8 };
    var out: [2]f64 = undefined;
    vdot_c128(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], 94.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -4.0, 1e-10);
}

test "vdot_c64 conjugate" {
    const testing = @import("std").testing;
    // conj(1+2i) * (3+4i) = 11-2i
    const a = [_]f32{ 1, 2 };
    const b = [_]f32{ 3, 4 };
    var out: [2]f32 = undefined;
    vdot_c64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 11.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-5);
}

test "vdot_c128 pure real vectors" {
    const testing = @import("std").testing;
    // conj(2+0i)*(3+0i) + conj(4+0i)*(5+0i) = 2*3 + 4*5 = 26 + 0i
    const a = [_]f64{ 2, 0, 4, 0 };
    const b = [_]f64{ 3, 0, 5, 0 };
    var out: [2]f64 = undefined;
    vdot_c128(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], 26.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
}

test "vdot_c128 pure imaginary vectors" {
    const testing = @import("std").testing;
    // conj(a)*b = (a_re*b_re + a_im*b_im) + (a_re*b_im - a_im*b_re)*i
    // = (0*0 + 2*3) + (0*3 - 2*0)*i = 6 + 0i
    const a = [_]f64{ 0, 2 };
    const b = [_]f64{ 0, 3 };
    var out: [2]f64 = undefined;
    vdot_c128(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 6.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
}

test "vdot_c128 self conjugate dot gives real result" {
    const testing = @import("std").testing;
    // conj(a) · a = sum |a_k|^2 (always real)
    // a = [(1+2i), (3+4i)]
    // |1+2i|^2 = 5, |3+4i|^2 = 25, total = 30
    const a = [_]f64{ 1, 2, 3, 4 };
    var out: [2]f64 = undefined;
    vdot_c128(&a, &a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 30.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
}

test "vdot_c64 two elements" {
    const testing = @import("std").testing;
    // Same as c128 two elements test but with f32
    // conj(1+2i)*(3+4i) + conj(5+6i)*(7+8i) = (11-2i) + (83-2i) = 94-4i
    const a = [_]f32{ 1, 2, 5, 6 };
    const b = [_]f32{ 3, 4, 7, 8 };
    var out: [2]f32 = undefined;
    vdot_c64(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], 94.0, 1e-3);
    try testing.expectApproxEqAbs(out[1], -4.0, 1e-3);
}

test "vdot_c64 self conjugate dot gives real result" {
    const testing = @import("std").testing;
    // a = [(1+1i), (2+2i), (3+3i)]
    // |1+1i|^2=2, |2+2i|^2=8, |3+3i|^2=18, total=28
    const a = [_]f32{ 1, 1, 2, 2, 3, 3 };
    var out: [2]f32 = undefined;
    vdot_c64(&a, &a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 28.0, 1e-3);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-3);
}

test "vdot_c128 zero vector" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 0, 0, 0, 0 };
    var out: [2]f64 = undefined;
    vdot_c128(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
}

test "vdot_c128 single real element" {
    const testing = @import("std").testing;
    // conj(5+0i)*(3+0i) = 15+0i
    const a = [_]f64{ 5, 0 };
    const b = [_]f64{ 3, 0 };
    var out: [2]f64 = undefined;
    vdot_c128(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 15.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
}

test "vdot_c64 pure imaginary" {
    const testing = @import("std").testing;
    // conj(a)*b = (a_re*b_re + a_im*b_im) + (a_re*b_im - a_im*b_re)*i
    // = (0*0 + -3*4) + (0*4 - -3*0)*i = -12 + 0i
    const a = [_]f32{ 0, 3 };
    const b = [_]f32{ 0, 4 };
    var out: [2]f32 = undefined;
    vdot_c64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 12.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-5);
}
