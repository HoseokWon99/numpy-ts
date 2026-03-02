// Linear algebra WASM kernels: matvec, vecmat, vecdot, outer, kron, cross, norm
//
// Uses native v128 widths: @Vector(2,f64) / @Vector(4,f32)
// Pointer-cast loads/stores for guaranteed v128.load/v128.store opcodes.

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

// ─── matvec: A[m×n] · x[n] → out[m] ────────────────────────────────────────

export fn matvec_f64(a: [*]const f64, x: [*]const f64, out: [*]f64, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    for (0..rows) |i| {
        var acc0: V2f64 = @splat(0.0);
        var acc1: V2f64 = @splat(0.0);
        const row = a + i * cols;
        var j: usize = 0;
        while (j + 4 <= cols) : (j += 4) {
            acc0 += load2_f64(row, j) * load2_f64(x, j);
            acc1 += load2_f64(row, j + 2) * load2_f64(x, j + 2);
        }
        while (j + 2 <= cols) : (j += 2) {
            acc0 += load2_f64(row, j) * load2_f64(x, j);
        }
        acc0 += acc1;
        var sum: f64 = acc0[0] + acc0[1];
        while (j < cols) : (j += 1) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

export fn matvec_f32(a: [*]const f32, x: [*]const f32, out: [*]f32, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    for (0..rows) |i| {
        var acc0: V4f32 = @splat(0.0);
        var acc1: V4f32 = @splat(0.0);
        const row = a + i * cols;
        var j: usize = 0;
        while (j + 8 <= cols) : (j += 8) {
            acc0 += load4_f32(row, j) * load4_f32(x, j);
            acc1 += load4_f32(row, j + 4) * load4_f32(x, j + 4);
        }
        while (j + 4 <= cols) : (j += 4) {
            acc0 += load4_f32(row, j) * load4_f32(x, j);
        }
        acc0 += acc1;
        var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
        while (j < cols) : (j += 1) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

// ─── vecmat: x[m] · A[m×n] → out[n] ────────────────────────────────────────

export fn vecmat_f64(x: [*]const f64, a: [*]const f64, out: [*]f64, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    // Zero output
    var j: usize = 0;
    while (j + 2 <= cols) : (j += 2) {
        store2_f64(out, j, @splat(0.0));
    }
    while (j < cols) : (j += 1) {
        out[j] = 0.0;
    }
    // Accumulate: out[j] += x[i] * A[i,j]
    for (0..rows) |i| {
        const xi: V2f64 = @splat(x[i]);
        const row = a + i * cols;
        j = 0;
        while (j + 4 <= cols) : (j += 4) {
            store2_f64(out, j, load2_f64(out, j) + xi * load2_f64(row, j));
            store2_f64(out, j + 2, load2_f64(out, j + 2) + xi * load2_f64(row, j + 2));
        }
        while (j + 2 <= cols) : (j += 2) {
            store2_f64(out, j, load2_f64(out, j) + xi * load2_f64(row, j));
        }
        while (j < cols) : (j += 1) {
            out[j] += x[i] * row[j];
        }
    }
}

export fn vecmat_f32(x: [*]const f32, a: [*]const f32, out: [*]f32, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    var j: usize = 0;
    while (j + 4 <= cols) : (j += 4) {
        store4_f32(out, j, @splat(0.0));
    }
    while (j < cols) : (j += 1) {
        out[j] = 0.0;
    }
    for (0..rows) |i| {
        const xi: V4f32 = @splat(x[i]);
        const row = a + i * cols;
        j = 0;
        while (j + 8 <= cols) : (j += 8) {
            store4_f32(out, j, load4_f32(out, j) + xi * load4_f32(row, j));
            store4_f32(out, j + 4, load4_f32(out, j + 4) + xi * load4_f32(row, j + 4));
        }
        while (j + 4 <= cols) : (j += 4) {
            store4_f32(out, j, load4_f32(out, j) + xi * load4_f32(row, j));
        }
        while (j < cols) : (j += 1) {
            out[j] += x[i] * row[j];
        }
    }
}

// ─── vecdot: batched dot products. a[batch×len], b[batch×len] → out[batch] ──

export fn vecdot_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, nbatch: u32, veclen: u32) void {
    const batch = @as(usize, nbatch);
    const len = @as(usize, veclen);
    for (0..batch) |bi| {
        const off = bi * len;
        var acc0: V2f64 = @splat(0.0);
        var acc1: V2f64 = @splat(0.0);
        var j: usize = 0;
        while (j + 4 <= len) : (j += 4) {
            acc0 += load2_f64(a, off + j) * load2_f64(b, off + j);
            acc1 += load2_f64(a, off + j + 2) * load2_f64(b, off + j + 2);
        }
        while (j + 2 <= len) : (j += 2) {
            acc0 += load2_f64(a, off + j) * load2_f64(b, off + j);
        }
        acc0 += acc1;
        var sum: f64 = acc0[0] + acc0[1];
        while (j < len) : (j += 1) {
            sum += a[off + j] * b[off + j];
        }
        out[bi] = sum;
    }
}

export fn vecdot_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, nbatch: u32, veclen: u32) void {
    const batch = @as(usize, nbatch);
    const len = @as(usize, veclen);
    for (0..batch) |bi| {
        const off = bi * len;
        var acc0: V4f32 = @splat(0.0);
        var acc1: V4f32 = @splat(0.0);
        var j: usize = 0;
        while (j + 8 <= len) : (j += 8) {
            acc0 += load4_f32(a, off + j) * load4_f32(b, off + j);
            acc1 += load4_f32(a, off + j + 4) * load4_f32(b, off + j + 4);
        }
        while (j + 4 <= len) : (j += 4) {
            acc0 += load4_f32(a, off + j) * load4_f32(b, off + j);
        }
        acc0 += acc1;
        var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
        while (j < len) : (j += 1) {
            sum += a[off + j] * b[off + j];
        }
        out[bi] = sum;
    }
}

// ─── outer: a[m] ⊗ b[n] → out[m×n] ────────────────────────────────────────

export fn outer_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    for (0..rows) |i| {
        const ai: V2f64 = @splat(a[i]);
        const row_out = out + i * cols;
        var j: usize = 0;
        while (j + 4 <= cols) : (j += 4) {
            store2_f64(row_out, j, ai * load2_f64(b, j));
            store2_f64(row_out, j + 2, ai * load2_f64(b, j + 2));
        }
        while (j + 2 <= cols) : (j += 2) {
            store2_f64(row_out, j, ai * load2_f64(b, j));
        }
        while (j < cols) : (j += 1) {
            row_out[j] = a[i] * b[j];
        }
    }
}

export fn outer_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    for (0..rows) |i| {
        const ai: V4f32 = @splat(a[i]);
        const row_out = out + i * cols;
        var j: usize = 0;
        while (j + 8 <= cols) : (j += 8) {
            store4_f32(row_out, j, ai * load4_f32(b, j));
            store4_f32(row_out, j + 4, ai * load4_f32(b, j + 4));
        }
        while (j + 4 <= cols) : (j += 4) {
            store4_f32(row_out, j, ai * load4_f32(b, j));
        }
        while (j < cols) : (j += 1) {
            row_out[j] = a[i] * b[j];
        }
    }
}

// ─── kron: Kronecker product A[am×an] ⊗ B[bm×bn] → out[(am*bm)×(an*bn)] ──

export fn kron_f64(
    a: [*]const f64, b: [*]const f64, out: [*]f64,
    am: u32, an: u32, bm: u32, bn: u32,
) void {
    const a_rows = @as(usize, am);
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..a_rows) |ia| {
        for (0..a_cols) |ja| {
            const aij: V2f64 = @splat(a[ia * a_cols + ja]);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;
                var jb: usize = 0;
                while (jb + 2 <= b_cols) : (jb += 2) {
                    store2_f64(out_row, jb, aij * load2_f64(b_row, jb));
                }
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] * b_row[jb];
                }
            }
        }
    }
}

export fn kron_f32(
    a: [*]const f32, b: [*]const f32, out: [*]f32,
    am: u32, an: u32, bm: u32, bn: u32,
) void {
    const a_rows = @as(usize, am);
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..a_rows) |ia| {
        for (0..a_cols) |ja| {
            const aij: V4f32 = @splat(a[ia * a_cols + ja]);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;
                var jb: usize = 0;
                while (jb + 4 <= b_cols) : (jb += 4) {
                    store4_f32(out_row, jb, aij * load4_f32(b_row, jb));
                }
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] * b_row[jb];
                }
            }
        }
    }
}

// ─── cross: cross product of n pairs of 3-vectors ──────────────────────────

export fn cross_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, n: u32) void {
    const count = @as(usize, n);
    for (0..count) |i| {
        const ao = i * 3;
        const bo = i * 3;
        const oo = i * 3;
        out[oo + 0] = a[ao + 1] * b[bo + 2] - a[ao + 2] * b[bo + 1];
        out[oo + 1] = a[ao + 2] * b[bo + 0] - a[ao + 0] * b[bo + 2];
        out[oo + 2] = a[ao + 0] * b[bo + 1] - a[ao + 1] * b[bo + 0];
    }
}

export fn cross_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) void {
    const count = @as(usize, n);
    for (0..count) |i| {
        const ao = i * 3;
        const bo = i * 3;
        const oo = i * 3;
        out[oo + 0] = a[ao + 1] * b[bo + 2] - a[ao + 2] * b[bo + 1];
        out[oo + 1] = a[ao + 2] * b[bo + 0] - a[ao + 0] * b[bo + 2];
        out[oo + 2] = a[ao + 0] * b[bo + 1] - a[ao + 1] * b[bo + 0];
    }
}

// ─── norm: L2 norm = sqrt(sum(x²)) ─────────────────────────────────────────

export fn norm_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    var acc0: V2f64 = @splat(0.0);
    var acc1: V2f64 = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v0 = load2_f64(ptr, i);
        const v1 = load2_f64(ptr, i + 2);
        acc0 += v0 * v0;
        acc1 += v1 * v1;
    }
    while (i + 2 <= len) : (i += 2) {
        const v = load2_f64(ptr, i);
        acc0 += v * v;
    }
    acc0 += acc1;
    var sum: f64 = acc0[0] + acc0[1];
    while (i < len) : (i += 1) {
        sum += ptr[i] * ptr[i];
    }
    return @sqrt(sum);
}

export fn norm_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    var acc0: V4f32 = @splat(0.0);
    var acc1: V4f32 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const v0 = load4_f32(ptr, i);
        const v1 = load4_f32(ptr, i + 4);
        acc0 += v0 * v0;
        acc1 += v1 * v1;
    }
    while (i + 4 <= len) : (i += 4) {
        const v = load4_f32(ptr, i);
        acc0 += v * v;
    }
    acc0 += acc1;
    var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
    while (i < len) : (i += 1) {
        sum += ptr[i] * ptr[i];
    }
    return @sqrt(sum);
}
