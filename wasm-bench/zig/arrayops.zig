// Array operation WASM kernels: roll, flip, tile, pad, take, gradient
//
// These are memory-movement-heavy operations that benefit from WASM
// primarily by avoiding JS interpreter overhead on element access.

const V2f64 = @Vector(2, f64);
const V4f32 = @Vector(4, f32);

inline fn load2_f64(ptr: [*]const f64, i: usize) V2f64 {
    return @as(*align(1) const V2f64, @ptrCast(ptr + i)).*;
}
inline fn store2_f64(ptr: [*]f64, i: usize, v: V2f64) void {
    @as(*align(1) V2f64, @ptrCast(ptr + i)).* = v;
}
inline fn load4_f32(ptr: [*]const f32, i: usize) V4f32 {
    return @as(*align(1) const V4f32, @ptrCast(ptr + i)).*;
}
inline fn store4_f32(ptr: [*]f32, i: usize, v: V4f32) void {
    @as(*align(1) V4f32, @ptrCast(ptr + i)).* = v;
}

// ─── SIMD memcpy helpers ────────────────────────────────────────────────────

fn simd_copy_f64(dst: [*]f64, src: [*]const f64, n: usize) void {
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        store2_f64(dst, i, load2_f64(src, i));
        store2_f64(dst, i + 2, load2_f64(src, i + 2));
    }
    while (i + 2 <= n) : (i += 2) {
        store2_f64(dst, i, load2_f64(src, i));
    }
    while (i < n) : (i += 1) {
        dst[i] = src[i];
    }
}

fn simd_copy_f32(dst: [*]f32, src: [*]const f32, n: usize) void {
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        store4_f32(dst, i, load4_f32(src, i));
        store4_f32(dst, i + 4, load4_f32(src, i + 4));
    }
    while (i + 4 <= n) : (i += 4) {
        store4_f32(dst, i, load4_f32(src, i));
    }
    while (i < n) : (i += 1) {
        dst[i] = src[i];
    }
}

fn simd_zero_f64(dst: [*]f64, n: usize) void {
    const zero: V2f64 = @splat(0.0);
    var i: usize = 0;
    while (i + 2 <= n) : (i += 2) {
        store2_f64(dst, i, zero);
    }
    while (i < n) : (i += 1) {
        dst[i] = 0.0;
    }
}

fn simd_zero_f32(dst: [*]f32, n: usize) void {
    const zero: V4f32 = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        store4_f32(dst, i, zero);
    }
    while (i < n) : (i += 1) {
        dst[i] = 0.0;
    }
}

// ─── roll: circular shift (1D) ──────────────────────────────────────────────
// out[i] = inp[(i - shift) mod n]  (positive shift = shift right)

export fn roll_f64(inp: [*]const f64, out: [*]f64, n: u32, shift: i32) void {
    const len = @as(usize, n);
    if (len == 0) return;
    // Normalize shift to [0, len)
    const s: usize = @intCast(@mod(@as(i64, shift), @as(i64, @intCast(len))));
    if (s == 0) {
        simd_copy_f64(out, inp, len);
        return;
    }
    // Copy last s elements to start, first (len-s) elements after
    simd_copy_f64(out, inp + (len - s), s);
    simd_copy_f64(out + s, inp, len - s);
}

export fn roll_f32(inp: [*]const f32, out: [*]f32, n: u32, shift: i32) void {
    const len = @as(usize, n);
    if (len == 0) return;
    const s: usize = @intCast(@mod(@as(i64, shift), @as(i64, @intCast(len))));
    if (s == 0) {
        simd_copy_f32(out, inp, len);
        return;
    }
    simd_copy_f32(out, inp + (len - s), s);
    simd_copy_f32(out + s, inp, len - s);
}

// ─── flip: reverse array (1D) ───────────────────────────────────────────────

export fn flip_f64(inp: [*]const f64, out: [*]f64, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| {
        out[i] = inp[len - 1 - i];
    }
}

export fn flip_f32(inp: [*]const f32, out: [*]f32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| {
        out[i] = inp[len - 1 - i];
    }
}

// ─── tile: repeat array `reps` times (1D) ──────────────────────────────────

export fn tile_f64(inp: [*]const f64, out: [*]f64, n: u32, reps: u32) void {
    const len = @as(usize, n);
    const r = @as(usize, reps);
    for (0..r) |rep| {
        simd_copy_f64(out + rep * len, inp, len);
    }
}

export fn tile_f32(inp: [*]const f32, out: [*]f32, n: u32, reps: u32) void {
    const len = @as(usize, n);
    const r = @as(usize, reps);
    for (0..r) |rep| {
        simd_copy_f32(out + rep * len, inp, len);
    }
}

// ─── pad: zero-pad 2D array [rows×cols] with `pw` on each side ──────────────
// Output is [(rows+2*pw) × (cols+2*pw)]

export fn pad_f64(
    inp: [*]const f64, out: [*]f64,
    rows: u32, cols: u32, pw: u32,
) void {
    const r = @as(usize, rows);
    const c = @as(usize, cols);
    const p = @as(usize, pw);
    const out_cols = c + 2 * p;
    const total = (r + 2 * p) * out_cols;

    // Zero entire output
    simd_zero_f64(out, total);

    // Copy input rows into padded positions
    for (0..r) |i| {
        simd_copy_f64(out + (i + p) * out_cols + p, inp + i * c, c);
    }
}

export fn pad_f32(
    inp: [*]const f32, out: [*]f32,
    rows: u32, cols: u32, pw: u32,
) void {
    const r = @as(usize, rows);
    const c = @as(usize, cols);
    const p = @as(usize, pw);
    const out_cols = c + 2 * p;
    const total = (r + 2 * p) * out_cols;

    simd_zero_f32(out, total);

    for (0..r) |i| {
        simd_copy_f32(out + (i + p) * out_cols + p, inp + i * c, c);
    }
}

// ─── take: gather elements by index (1D) ────────────────────────────────────
// out[i] = data[indices[i]]

export fn take_f64(data: [*]const f64, indices: [*]const u32, out: [*]f64, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| {
        out[i] = data[@as(usize, indices[i])];
    }
}

export fn take_f32(data: [*]const f32, indices: [*]const u32, out: [*]f32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| {
        out[i] = data[@as(usize, indices[i])];
    }
}

// ─── gradient: numerical gradient (central differences, 1D) ─────────────────
// out[0] = x[1] - x[0], out[n-1] = x[n-1] - x[n-2], interior = (x[i+1] - x[i-1]) / 2

export fn gradient_f64(inp: [*]const f64, out: [*]f64, n: u32) void {
    const len = @as(usize, n);
    if (len < 2) return;
    // Boundaries
    out[0] = inp[1] - inp[0];
    out[len - 1] = inp[len - 1] - inp[len - 2];
    // Interior: central differences with SIMD
    if (len <= 2) return;
    const half: V2f64 = @splat(0.5);
    var i: usize = 1;
    while (i + 2 < len) : (i += 2) {
        const fwd = load2_f64(inp, i + 1);
        const bwd = load2_f64(inp, i - 1);
        store2_f64(out, i, (fwd - bwd) * half);
    }
    while (i < len - 1) : (i += 1) {
        out[i] = (inp[i + 1] - inp[i - 1]) * 0.5;
    }
}

export fn gradient_f32(inp: [*]const f32, out: [*]f32, n: u32) void {
    const len = @as(usize, n);
    if (len < 2) return;
    out[0] = inp[1] - inp[0];
    out[len - 1] = inp[len - 1] - inp[len - 2];
    if (len <= 2) return;
    const half: V4f32 = @splat(0.5);
    var i: usize = 1;
    while (i + 4 < len) : (i += 4) {
        const fwd = load4_f32(inp, i + 1);
        const bwd = load4_f32(inp, i - 1);
        store4_f32(out, i, (fwd - bwd) * half);
    }
    while (i < len - 1) : (i += 1) {
        out[i] = (inp[i + 1] - inp[i - 1]) * 0.5;
    }
}
