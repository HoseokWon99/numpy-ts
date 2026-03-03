// WASM reduction kernels for f32/f64 with SIMD
//
// Uses native v128 widths: @Vector(2,f64) / @Vector(4,f32)
// Two accumulators for sum/prod to saturate memory bandwidth.
// Pointer-cast loads to guarantee v128.load opcodes.

const V2f64 = @Vector(2, f64);
const V4f32 = @Vector(4, f32);

inline fn load2_f64(ptr: [*]const f64, i: usize) V2f64 {
    return @as(*align(1) const V2f64, @ptrCast(ptr + i)).*;
}
inline fn load4_f32(ptr: [*]const f32, i: usize) V4f32 {
    return @as(*align(1) const V4f32, @ptrCast(ptr + i)).*;
}
inline fn store2_f64(ptr: [*]f64, i: usize, v: V2f64) void {
    @as(*align(1) V2f64, @ptrCast(ptr + i)).* = v;
}
inline fn store4_f32(ptr: [*]f32, i: usize, v: V4f32) void {
    @as(*align(1) V4f32, @ptrCast(ptr + i)).* = v;
}

// ─── f64 reductions ─────────────────────────────────────────────────────────

export fn sum_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    var acc0: V2f64 = @splat(0.0);
    var acc1: V2f64 = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        acc0 += load2_f64(ptr, i);
        acc1 += load2_f64(ptr, i + 2);
    }
    while (i + 2 <= len) : (i += 2) {
        acc0 += load2_f64(ptr, i);
    }
    acc0 += acc1;
    var result: f64 = acc0[0] + acc0[1];
    while (i < len) : (i += 1) {
        result += ptr[i];
    }
    return result;
}

export fn max_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    if (len == 0) return -@as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    var acc: V2f64 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 2 <= len) : (i += 2) {
        const v = load2_f64(ptr, i);
        acc = @select(f64, v > acc, v, acc);
    }
    var result: f64 = if (acc[0] > acc[1]) acc[0] else acc[1];
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }
    return result;
}

export fn min_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    if (len == 0) return @as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    var acc: V2f64 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 2 <= len) : (i += 2) {
        const v = load2_f64(ptr, i);
        acc = @select(f64, v < acc, v, acc);
    }
    var result: f64 = if (acc[0] < acc[1]) acc[0] else acc[1];
    while (i < len) : (i += 1) {
        if (ptr[i] < result) result = ptr[i];
    }
    return result;
}

export fn prod_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    var acc0: V2f64 = @splat(1.0);
    var acc1: V2f64 = @splat(1.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        acc0 *= load2_f64(ptr, i);
        acc1 *= load2_f64(ptr, i + 2);
    }
    while (i + 2 <= len) : (i += 2) {
        acc0 *= load2_f64(ptr, i);
    }
    acc0 *= acc1;
    var result: f64 = acc0[0] * acc0[1];
    while (i < len) : (i += 1) {
        result *= ptr[i];
    }
    return result;
}

export fn mean_f64(ptr: [*]const f64, n: u32) f64 {
    return sum_f64(ptr, n) / @as(f64, @floatFromInt(n));
}

// ─── f32 reductions ─────────────────────────────────────────────────────────

export fn sum_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    var acc0: V4f32 = @splat(0.0);
    var acc1: V4f32 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        acc0 += load4_f32(ptr, i);
        acc1 += load4_f32(ptr, i + 4);
    }
    while (i + 4 <= len) : (i += 4) {
        acc0 += load4_f32(ptr, i);
    }
    acc0 += acc1;
    var result: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
    while (i < len) : (i += 1) {
        result += ptr[i];
    }
    return result;
}

export fn max_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    if (len == 0) return -@as(f32, @bitCast(@as(u32, 0x7F800000)));
    var acc: V4f32 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v = load4_f32(ptr, i);
        acc = @select(f32, v > acc, v, acc);
    }
    var result: f32 = acc[0];
    if (acc[1] > result) result = acc[1];
    if (acc[2] > result) result = acc[2];
    if (acc[3] > result) result = acc[3];
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }
    return result;
}

export fn min_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    if (len == 0) return @as(f32, @bitCast(@as(u32, 0x7F800000)));
    var acc: V4f32 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v = load4_f32(ptr, i);
        acc = @select(f32, v < acc, v, acc);
    }
    var result: f32 = acc[0];
    if (acc[1] < result) result = acc[1];
    if (acc[2] < result) result = acc[2];
    if (acc[3] < result) result = acc[3];
    while (i < len) : (i += 1) {
        if (ptr[i] < result) result = ptr[i];
    }
    return result;
}

export fn prod_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    var acc0: V4f32 = @splat(1.0);
    var acc1: V4f32 = @splat(1.0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        acc0 *= load4_f32(ptr, i);
        acc1 *= load4_f32(ptr, i + 4);
    }
    while (i + 4 <= len) : (i += 4) {
        acc0 *= load4_f32(ptr, i);
    }
    acc0 *= acc1;
    var result: f32 = acc0[0] * acc0[1] * acc0[2] * acc0[3];
    while (i < len) : (i += 1) {
        result *= ptr[i];
    }
    return result;
}

export fn mean_f32(ptr: [*]const f32, n: u32) f32 {
    return sum_f32(ptr, n) / @as(f32, @floatFromInt(n));
}

// ─── nanmax: max ignoring NaN ───────────────────────────────────────────────

export fn nanmax_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    if (len == 0) return -@as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    // Find first non-NaN
    var start: usize = 0;
    while (start < len and ptr[start] != ptr[start]) : (start += 1) {}
    if (start == len) return ptr[0]; // all NaN, return NaN
    var result: f64 = ptr[start];
    var i = start + 1;
    while (i < len) : (i += 1) {
        const v = ptr[i];
        if (v == v and v > result) result = v;
    }
    return result;
}

export fn nanmax_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    if (len == 0) return -@as(f32, @bitCast(@as(u32, 0x7F800000)));
    var start: usize = 0;
    while (start < len and ptr[start] != ptr[start]) : (start += 1) {}
    if (start == len) return ptr[0];
    var result: f32 = ptr[start];
    var i = start + 1;
    while (i < len) : (i += 1) {
        const v = ptr[i];
        if (v == v and v > result) result = v;
    }
    return result;
}

// ─── nanmin: min ignoring NaN ───────────────────────────────────────────────

export fn nanmin_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    if (len == 0) return @as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    var start: usize = 0;
    while (start < len and ptr[start] != ptr[start]) : (start += 1) {}
    if (start == len) return ptr[0];
    var result: f64 = ptr[start];
    var i = start + 1;
    while (i < len) : (i += 1) {
        const v = ptr[i];
        if (v == v and v < result) result = v;
    }
    return result;
}

export fn nanmin_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    if (len == 0) return @as(f32, @bitCast(@as(u32, 0x7F800000)));
    var start: usize = 0;
    while (start < len and ptr[start] != ptr[start]) : (start += 1) {}
    if (start == len) return ptr[0];
    var result: f32 = ptr[start];
    var i = start + 1;
    while (i < len) : (i += 1) {
        const v = ptr[i];
        if (v == v and v < result) result = v;
    }
    return result;
}

// ─── diff: first-order differences ──────────────────────────────────────────
// out[i] = in[i+1] - in[i], output has n-1 elements

export fn diff_f64(in_ptr: [*]const f64, out_ptr: [*]f64, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    const out_len = len - 1;
    var i: usize = 0;
    while (i + 2 <= out_len) : (i += 2) {
        const v0 = load2_f64(in_ptr, i);
        const v1 = load2_f64(in_ptr, i + 1);
        // v1 - v0 gives [in[i+1]-in[i], in[i+2]-in[i+1]]
        store2_f64(out_ptr, i, v1 - v0);
    }
    while (i < out_len) : (i += 1) {
        out_ptr[i] = in_ptr[i + 1] - in_ptr[i];
    }
}

export fn diff_f32(in_ptr: [*]const f32, out_ptr: [*]f32, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    const out_len = len - 1;
    var i: usize = 0;
    while (i + 4 <= out_len) : (i += 4) {
        const v0 = load4_f32(in_ptr, i);
        const v1 = load4_f32(in_ptr, i + 1);
        store4_f32(out_ptr, i, v1 - v0);
    }
    while (i < out_len) : (i += 1) {
        out_ptr[i] = in_ptr[i + 1] - in_ptr[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER REDUCTIONS (i32, i16, i8) — widened accumulators for i16/i8
// ═══════════════════════════════════════════════════════════════════════════

const V4i32 = @Vector(4, i32);

inline fn load4_i32(ptr: [*]const i32, i: usize) V4i32 {
    return @as(*align(1) const V4i32, @ptrCast(ptr + i)).*;
}

// ─── i32 reductions ──────────────────────────────────────────────────────

export fn sum_i32(ptr: [*]const i32, n: u32) i32 {
    const len = @as(usize, n);
    var acc0: V4i32 = @splat(0);
    var acc1: V4i32 = @splat(0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        acc0 +%= load4_i32(ptr, i);
        acc1 +%= load4_i32(ptr, i + 4);
    }
    while (i + 4 <= len) : (i += 4) {
        acc0 +%= load4_i32(ptr, i);
    }
    acc0 +%= acc1;
    var result: i32 = acc0[0] +% acc0[1] +% acc0[2] +% acc0[3];
    while (i < len) : (i += 1) {
        result +%= ptr[i];
    }
    return result;
}

export fn max_i32(ptr: [*]const i32, n: u32) i32 {
    const len = @as(usize, n);
    if (len == 0) return -2147483648; // i32 min
    var acc: V4i32 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        acc = @max(acc, load4_i32(ptr, i));
    }
    var result: i32 = @max(acc[0], @max(acc[1], @max(acc[2], acc[3])));
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }
    return result;
}

export fn min_i32(ptr: [*]const i32, n: u32) i32 {
    const len = @as(usize, n);
    if (len == 0) return 2147483647; // i32 max
    var acc: V4i32 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        acc = @min(acc, load4_i32(ptr, i));
    }
    var result: i32 = @min(acc[0], @min(acc[1], @min(acc[2], acc[3])));
    while (i < len) : (i += 1) {
        if (ptr[i] < result) result = ptr[i];
    }
    return result;
}

// ─── i16 reductions (widen to i32 accumulators) ──────────────────────────

export fn sum_i16(ptr: [*]const i16, n: u32) i32 {
    const len = @as(usize, n);
    var result: i32 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        result +%= @as(i32, ptr[i]);
    }
    return result;
}

export fn max_i16(ptr: [*]const i16, n: u32) i16 {
    const len = @as(usize, n);
    if (len == 0) return -32768;
    var result: i16 = ptr[0];
    var i: usize = 1;
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }
    return result;
}

export fn min_i16(ptr: [*]const i16, n: u32) i16 {
    const len = @as(usize, n);
    if (len == 0) return 32767;
    var result: i16 = ptr[0];
    var i: usize = 1;
    while (i < len) : (i += 1) {
        if (ptr[i] < result) result = ptr[i];
    }
    return result;
}

// ─── i8 reductions (widen to i32 accumulators) ───────────────────────────

export fn sum_i8(ptr: [*]const i8, n: u32) i32 {
    const len = @as(usize, n);
    var result: i32 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        result +%= @as(i32, ptr[i]);
    }
    return result;
}

export fn max_i8(ptr: [*]const i8, n: u32) i8 {
    const len = @as(usize, n);
    if (len == 0) return -128;
    var result: i8 = ptr[0];
    var i: usize = 1;
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }
    return result;
}

export fn min_i8(ptr: [*]const i8, n: u32) i8 {
    const len = @as(usize, n);
    if (len == 0) return 127;
    var result: i8 = ptr[0];
    var i: usize = 1;
    while (i < len) : (i += 1) {
        if (ptr[i] < result) result = ptr[i];
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX REDUCTIONS (c128, c64)
// ═══════════════════════════════════════════════════════════════════════════

// sum_c128: sum of n complex numbers (2n f64s), writes 2 f64s to out
export fn sum_c128(ptr: [*]const f64, out: [*]f64, n: u32) void {
    // Treat as f64 sum on 2*n elements, accumulate re and im in V2f64
    const len = @as(usize, n);
    var acc0: V2f64 = @splat(0.0);
    var acc1: V2f64 = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        acc0 += load2_f64(ptr, i * 2);
        acc0 += load2_f64(ptr, (i + 1) * 2);
        acc1 += load2_f64(ptr, (i + 2) * 2);
        acc1 += load2_f64(ptr, (i + 3) * 2);
    }
    while (i < len) : (i += 1) {
        acc0 += load2_f64(ptr, i * 2);
    }
    acc0 += acc1;
    out[0] = acc0[0];
    out[1] = acc0[1];
}

// sum_c64: sum of n complex numbers (2n f32s), writes 2 f32s to out
export fn sum_c64(ptr: [*]const f32, out: [*]f32, n: u32) void {
    const len = @as(usize, n);
    var re_sum: f32 = 0.0;
    var im_sum: f32 = 0.0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        re_sum += ptr[2 * i];
        im_sum += ptr[2 * i + 1];
    }
    out[0] = re_sum;
    out[1] = im_sum;
}
