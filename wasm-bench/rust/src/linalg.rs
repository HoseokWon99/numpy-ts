// Linear algebra kernels: matvec, vecmat, vecdot, outer, kron, cross, norm

use core::arch::wasm32::*;

// ─── matvec: A[m×n] · x[n] → out[m] ────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn matvec_f64(a: *const f64, x: *const f64, out: *mut f64, m: u32, n: u32) {
    let rows = m as usize;
    let cols = n as usize;
    for i in 0..rows {
        let row = a.add(i * cols);
        let mut acc0 = f64x2_splat(0.0);
        let mut acc1 = f64x2_splat(0.0);
        let mut j = 0;
        while j + 4 <= cols {
            acc0 = f64x2_add(acc0, f64x2_mul(
                v128_load(row.add(j) as *const v128),
                v128_load(x.add(j) as *const v128),
            ));
            acc1 = f64x2_add(acc1, f64x2_mul(
                v128_load(row.add(j + 2) as *const v128),
                v128_load(x.add(j + 2) as *const v128),
            ));
            j += 4;
        }
        while j + 2 <= cols {
            acc0 = f64x2_add(acc0, f64x2_mul(
                v128_load(row.add(j) as *const v128),
                v128_load(x.add(j) as *const v128),
            ));
            j += 2;
        }
        acc0 = f64x2_add(acc0, acc1);
        let mut sum = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
        while j < cols { sum += *row.add(j) * *x.add(j); j += 1; }
        *out.add(i) = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn matvec_f32(a: *const f32, x: *const f32, out: *mut f32, m: u32, n: u32) {
    let rows = m as usize;
    let cols = n as usize;
    for i in 0..rows {
        let row = a.add(i * cols);
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut j = 0;
        while j + 8 <= cols {
            acc0 = f32x4_add(acc0, f32x4_mul(
                v128_load(row.add(j) as *const v128),
                v128_load(x.add(j) as *const v128),
            ));
            acc1 = f32x4_add(acc1, f32x4_mul(
                v128_load(row.add(j + 4) as *const v128),
                v128_load(x.add(j + 4) as *const v128),
            ));
            j += 8;
        }
        while j + 4 <= cols {
            acc0 = f32x4_add(acc0, f32x4_mul(
                v128_load(row.add(j) as *const v128),
                v128_load(x.add(j) as *const v128),
            ));
            j += 4;
        }
        acc0 = f32x4_add(acc0, acc1);
        let mut sum = f32x4_extract_lane::<0>(acc0) + f32x4_extract_lane::<1>(acc0)
            + f32x4_extract_lane::<2>(acc0) + f32x4_extract_lane::<3>(acc0);
        while j < cols { sum += *row.add(j) * *x.add(j); j += 1; }
        *out.add(i) = sum;
    }
}

// ─── vecmat: x[m] · A[m×n] → out[n] ────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vecmat_f64(x: *const f64, a: *const f64, out: *mut f64, m: u32, n: u32) {
    let rows = m as usize;
    let cols = n as usize;
    // Zero output
    let mut j = 0;
    while j + 2 <= cols {
        v128_store(out.add(j) as *mut v128, f64x2_splat(0.0));
        j += 2;
    }
    while j < cols { *out.add(j) = 0.0; j += 1; }
    // Accumulate
    for i in 0..rows {
        let xi = f64x2_splat(*x.add(i));
        let row = a.add(i * cols);
        j = 0;
        while j + 4 <= cols {
            v128_store(out.add(j) as *mut v128, f64x2_add(
                v128_load(out.add(j) as *const v128),
                f64x2_mul(xi, v128_load(row.add(j) as *const v128)),
            ));
            v128_store(out.add(j + 2) as *mut v128, f64x2_add(
                v128_load(out.add(j + 2) as *const v128),
                f64x2_mul(xi, v128_load(row.add(j + 2) as *const v128)),
            ));
            j += 4;
        }
        while j + 2 <= cols {
            v128_store(out.add(j) as *mut v128, f64x2_add(
                v128_load(out.add(j) as *const v128),
                f64x2_mul(xi, v128_load(row.add(j) as *const v128)),
            ));
            j += 2;
        }
        while j < cols { *out.add(j) += *x.add(i) * *row.add(j); j += 1; }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vecmat_f32(x: *const f32, a: *const f32, out: *mut f32, m: u32, n: u32) {
    let rows = m as usize;
    let cols = n as usize;
    let mut j = 0;
    while j + 4 <= cols {
        v128_store(out.add(j) as *mut v128, f32x4_splat(0.0));
        j += 4;
    }
    while j < cols { *out.add(j) = 0.0; j += 1; }
    for i in 0..rows {
        let xi = f32x4_splat(*x.add(i));
        let row = a.add(i * cols);
        j = 0;
        while j + 8 <= cols {
            v128_store(out.add(j) as *mut v128, f32x4_add(
                v128_load(out.add(j) as *const v128),
                f32x4_mul(xi, v128_load(row.add(j) as *const v128)),
            ));
            v128_store(out.add(j + 4) as *mut v128, f32x4_add(
                v128_load(out.add(j + 4) as *const v128),
                f32x4_mul(xi, v128_load(row.add(j + 4) as *const v128)),
            ));
            j += 8;
        }
        while j + 4 <= cols {
            v128_store(out.add(j) as *mut v128, f32x4_add(
                v128_load(out.add(j) as *const v128),
                f32x4_mul(xi, v128_load(row.add(j) as *const v128)),
            ));
            j += 4;
        }
        while j < cols { *out.add(j) += *x.add(i) * *row.add(j); j += 1; }
    }
}

// ─── vecdot: batched dot products ───────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vecdot_f64(
    a: *const f64, b: *const f64, out: *mut f64, nbatch: u32, veclen: u32,
) {
    let batch = nbatch as usize;
    let len = veclen as usize;
    for bi in 0..batch {
        let off = bi * len;
        let mut acc0 = f64x2_splat(0.0);
        let mut acc1 = f64x2_splat(0.0);
        let mut j = 0;
        while j + 4 <= len {
            acc0 = f64x2_add(acc0, f64x2_mul(
                v128_load(a.add(off + j) as *const v128),
                v128_load(b.add(off + j) as *const v128),
            ));
            acc1 = f64x2_add(acc1, f64x2_mul(
                v128_load(a.add(off + j + 2) as *const v128),
                v128_load(b.add(off + j + 2) as *const v128),
            ));
            j += 4;
        }
        while j + 2 <= len {
            acc0 = f64x2_add(acc0, f64x2_mul(
                v128_load(a.add(off + j) as *const v128),
                v128_load(b.add(off + j) as *const v128),
            ));
            j += 2;
        }
        acc0 = f64x2_add(acc0, acc1);
        let mut sum = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
        while j < len { sum += *a.add(off + j) * *b.add(off + j); j += 1; }
        *out.add(bi) = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn vecdot_f32(
    a: *const f32, b: *const f32, out: *mut f32, nbatch: u32, veclen: u32,
) {
    let batch = nbatch as usize;
    let len = veclen as usize;
    for bi in 0..batch {
        let off = bi * len;
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut j = 0;
        while j + 8 <= len {
            acc0 = f32x4_add(acc0, f32x4_mul(
                v128_load(a.add(off + j) as *const v128),
                v128_load(b.add(off + j) as *const v128),
            ));
            acc1 = f32x4_add(acc1, f32x4_mul(
                v128_load(a.add(off + j + 4) as *const v128),
                v128_load(b.add(off + j + 4) as *const v128),
            ));
            j += 8;
        }
        while j + 4 <= len {
            acc0 = f32x4_add(acc0, f32x4_mul(
                v128_load(a.add(off + j) as *const v128),
                v128_load(b.add(off + j) as *const v128),
            ));
            j += 4;
        }
        acc0 = f32x4_add(acc0, acc1);
        let mut sum = f32x4_extract_lane::<0>(acc0) + f32x4_extract_lane::<1>(acc0)
            + f32x4_extract_lane::<2>(acc0) + f32x4_extract_lane::<3>(acc0);
        while j < len { sum += *a.add(off + j) * *b.add(off + j); j += 1; }
        *out.add(bi) = sum;
    }
}

// ─── outer: a[m] ⊗ b[n] → out[m×n] ────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn outer_f64(
    a: *const f64, b: *const f64, out: *mut f64, m: u32, n: u32,
) {
    let rows = m as usize;
    let cols = n as usize;
    for i in 0..rows {
        let ai = f64x2_splat(*a.add(i));
        let row_out = out.add(i * cols);
        let mut j = 0;
        while j + 4 <= cols {
            v128_store(row_out.add(j) as *mut v128, f64x2_mul(ai, v128_load(b.add(j) as *const v128)));
            v128_store(row_out.add(j + 2) as *mut v128, f64x2_mul(ai, v128_load(b.add(j + 2) as *const v128)));
            j += 4;
        }
        while j + 2 <= cols {
            v128_store(row_out.add(j) as *mut v128, f64x2_mul(ai, v128_load(b.add(j) as *const v128)));
            j += 2;
        }
        while j < cols { *row_out.add(j) = *a.add(i) * *b.add(j); j += 1; }
    }
}

#[no_mangle]
pub unsafe extern "C" fn outer_f32(
    a: *const f32, b: *const f32, out: *mut f32, m: u32, n: u32,
) {
    let rows = m as usize;
    let cols = n as usize;
    for i in 0..rows {
        let ai = f32x4_splat(*a.add(i));
        let row_out = out.add(i * cols);
        let mut j = 0;
        while j + 8 <= cols {
            v128_store(row_out.add(j) as *mut v128, f32x4_mul(ai, v128_load(b.add(j) as *const v128)));
            v128_store(row_out.add(j + 4) as *mut v128, f32x4_mul(ai, v128_load(b.add(j + 4) as *const v128)));
            j += 8;
        }
        while j + 4 <= cols {
            v128_store(row_out.add(j) as *mut v128, f32x4_mul(ai, v128_load(b.add(j) as *const v128)));
            j += 4;
        }
        while j < cols { *row_out.add(j) = *a.add(i) * *b.add(j); j += 1; }
    }
}

// ─── kron: Kronecker product ────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn kron_f64(
    a: *const f64, b: *const f64, out: *mut f64,
    am: u32, an: u32, bm: u32, bn: u32,
) {
    let (ar, ac, br, bc) = (am as usize, an as usize, bm as usize, bn as usize);
    let out_cols = ac * bc;
    for ia in 0..ar {
        for ja in 0..ac {
            let aij = f64x2_splat(*a.add(ia * ac + ja));
            for ib in 0..br {
                let out_row = out.add((ia * br + ib) * out_cols + ja * bc);
                let b_row = b.add(ib * bc);
                let mut jb = 0;
                while jb + 2 <= bc {
                    v128_store(out_row.add(jb) as *mut v128, f64x2_mul(aij, v128_load(b_row.add(jb) as *const v128)));
                    jb += 2;
                }
                while jb < bc { *out_row.add(jb) = *a.add(ia * ac + ja) * *b_row.add(jb); jb += 1; }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn kron_f32(
    a: *const f32, b: *const f32, out: *mut f32,
    am: u32, an: u32, bm: u32, bn: u32,
) {
    let (ar, ac, br, bc) = (am as usize, an as usize, bm as usize, bn as usize);
    let out_cols = ac * bc;
    for ia in 0..ar {
        for ja in 0..ac {
            let aij = f32x4_splat(*a.add(ia * ac + ja));
            for ib in 0..br {
                let out_row = out.add((ia * br + ib) * out_cols + ja * bc);
                let b_row = b.add(ib * bc);
                let mut jb = 0;
                while jb + 4 <= bc {
                    v128_store(out_row.add(jb) as *mut v128, f32x4_mul(aij, v128_load(b_row.add(jb) as *const v128)));
                    jb += 4;
                }
                while jb < bc { *out_row.add(jb) = *a.add(ia * ac + ja) * *b_row.add(jb); jb += 1; }
            }
        }
    }
}

// ─── cross: cross product of n pairs of 3-vectors ──────────────────────────

#[no_mangle]
pub unsafe extern "C" fn cross_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    for i in 0..n as usize {
        let (ao, bo, oo) = (i * 3, i * 3, i * 3);
        *out.add(oo) = *a.add(ao + 1) * *b.add(bo + 2) - *a.add(ao + 2) * *b.add(bo + 1);
        *out.add(oo + 1) = *a.add(ao + 2) * *b.add(bo) - *a.add(ao) * *b.add(bo + 2);
        *out.add(oo + 2) = *a.add(ao) * *b.add(bo + 1) - *a.add(ao + 1) * *b.add(bo);
    }
}

#[no_mangle]
pub unsafe extern "C" fn cross_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    for i in 0..n as usize {
        let (ao, bo, oo) = (i * 3, i * 3, i * 3);
        *out.add(oo) = *a.add(ao + 1) * *b.add(bo + 2) - *a.add(ao + 2) * *b.add(bo + 1);
        *out.add(oo + 1) = *a.add(ao + 2) * *b.add(bo) - *a.add(ao) * *b.add(bo + 2);
        *out.add(oo + 2) = *a.add(ao) * *b.add(bo + 1) - *a.add(ao + 1) * *b.add(bo);
    }
}

// ─── norm: L2 norm ──────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn norm_f64(ptr: *const f64, n: u32) -> f64 {
    let len = n as usize;
    let mut acc0 = f64x2_splat(0.0);
    let mut acc1 = f64x2_splat(0.0);
    let mut i = 0;
    while i + 4 <= len {
        let v0 = v128_load(ptr.add(i) as *const v128);
        let v1 = v128_load(ptr.add(i + 2) as *const v128);
        acc0 = f64x2_add(acc0, f64x2_mul(v0, v0));
        acc1 = f64x2_add(acc1, f64x2_mul(v1, v1));
        i += 4;
    }
    while i + 2 <= len {
        let v = v128_load(ptr.add(i) as *const v128);
        acc0 = f64x2_add(acc0, f64x2_mul(v, v));
        i += 2;
    }
    acc0 = f64x2_add(acc0, acc1);
    let mut sum = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
    while i < len { let v = *ptr.add(i); sum += v * v; i += 1; }
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(sum)))
}

#[no_mangle]
pub unsafe extern "C" fn norm_f32(ptr: *const f32, n: u32) -> f32 {
    let len = n as usize;
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0;
    while i + 8 <= len {
        let v0 = v128_load(ptr.add(i) as *const v128);
        let v1 = v128_load(ptr.add(i + 4) as *const v128);
        acc0 = f32x4_add(acc0, f32x4_mul(v0, v0));
        acc1 = f32x4_add(acc1, f32x4_mul(v1, v1));
        i += 8;
    }
    while i + 4 <= len {
        let v = v128_load(ptr.add(i) as *const v128);
        acc0 = f32x4_add(acc0, f32x4_mul(v, v));
        i += 4;
    }
    acc0 = f32x4_add(acc0, acc1);
    let mut sum = f32x4_extract_lane::<0>(acc0) + f32x4_extract_lane::<1>(acc0)
        + f32x4_extract_lane::<2>(acc0) + f32x4_extract_lane::<3>(acc0);
    while i < len { let v = *ptr.add(i); sum += v * v; i += 1; }
    f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(sum)))
}

// ─── Internal tiled matmul (duplicated from lib.rs for linalg module) ───────

const TILE_INT: usize = 48;

unsafe fn matmul_internal_f64(a: *const f64, b: *const f64, c: *mut f64, m: usize, n: usize, k: usize) {
    for i in 0..m * n { *c.add(i) = 0.0; }
    let mut ii = 0;
    while ii < m {
        let ie = if ii + TILE_INT < m { ii + TILE_INT } else { m };
        let mut kk = 0;
        while kk < k {
            let ke = if kk + TILE_INT < k { kk + TILE_INT } else { k };
            let mut jj = 0;
            while jj < n {
                let je = if jj + TILE_INT < n { jj + TILE_INT } else { n };
                let mut ri = ii;
                while ri < ie {
                    let mut rk = kk;
                    while rk < ke {
                        let aik = *a.add(ri * k + rk);
                        let mut j = jj;
                        while j < je {
                            *c.add(ri * n + j) += aik * *b.add(rk * n + j);
                            j += 1;
                        }
                        rk += 1;
                    }
                    ri += 1;
                }
                jj += TILE_INT;
            }
            kk += TILE_INT;
        }
        ii += TILE_INT;
    }
}

unsafe fn copy_f64(dst: *mut f64, src: *const f64, len: usize) {
    for i in 0..len { *dst.add(i) = *src.add(i); }
}

fn sqrt_f64_scalar(x: f64) -> f64 {
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
}

// ─── matrix_power: out = a^power via binary exponentiation ──────────────────

#[no_mangle]
pub unsafe extern "C" fn matrix_power_f64(a: *const f64, out: *mut f64, scratch: *mut f64, n: u32, power: u32) {
    let nn = n as usize;
    let sz = nn * nn;
    let cur = scratch;
    let tmp = scratch.add(sz);
    let mut p = power as usize;

    for i in 0..sz { *out.add(i) = 0.0; }
    for i in 0..nn { *out.add(i * nn + i) = 1.0; }
    copy_f64(cur, a, sz);

    while p > 0 {
        if p & 1 != 0 {
            matmul_internal_f64(out as *const f64, cur as *const f64, tmp, nn, nn, nn);
            copy_f64(out, tmp as *const f64, sz);
        }
        p >>= 1;
        if p > 0 {
            matmul_internal_f64(cur as *const f64, cur as *const f64, tmp, nn, nn, nn);
            copy_f64(cur, tmp as *const f64, sz);
        }
    }
}

// ─── multi_dot3: out = a @ b @ c ────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn multi_dot3_f64(a: *const f64, b: *const f64, c: *const f64, out: *mut f64, tmp: *mut f64, n: u32) {
    let nn = n as usize;
    matmul_internal_f64(a, b, tmp, nn, nn, nn);
    matmul_internal_f64(tmp as *const f64, c, out, nn, nn, nn);
}

// ─── qr: Householder QR decomposition ──────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn qr_f64(a: *mut f64, q: *mut f64, r: *mut f64, tau: *mut f64, _scratch: *mut f64, m: u32, n: u32) {
    let rows = m as usize;
    let cols = n as usize;
    let k = if rows < cols { rows } else { cols };

    for j in 0..k {
        let mut norm_sq = 0.0f64;
        for i in j..rows { let v = *a.add(i * cols + j); norm_sq += v * v; }
        let mut nrm = sqrt_f64_scalar(norm_sq);
        if nrm == 0.0 { *tau.add(j) = 0.0; continue; }

        let ajj = *a.add(j * cols + j);
        if ajj >= 0.0 { nrm = -nrm; }
        let alpha = nrm;

        *a.add(j * cols + j) -= alpha;
        let v0 = *a.add(j * cols + j);

        let mut vtv = v0 * v0;
        for i in (j + 1)..rows { let vi = *a.add(i * cols + j); vtv += vi * vi; }
        if vtv == 0.0 {
            *tau.add(j) = 0.0;
            *a.add(j * cols + j) = alpha;
            continue;
        }
        *tau.add(j) = 2.0 / vtv;

        for col in (j + 1)..cols {
            let mut dot = 0.0f64;
            for i in j..rows { dot += *a.add(i * cols + j) * *a.add(i * cols + col); }
            let factor = *tau.add(j) * dot;
            for i in j..rows { *a.add(i * cols + col) -= factor * *a.add(i * cols + j); }
        }

        *a.add(j * cols + j) = alpha;
    }

    // Extract R
    for i in 0..k {
        for j in 0..cols {
            *r.add(i * cols + j) = if j >= i { *a.add(i * cols + j) } else { 0.0 };
        }
    }

    // Reconstruct Q
    for i in 0..rows * k { *q.add(i) = 0.0; }
    for i in 0..k { *q.add(i * k + i) = 1.0; }

    let mut jrev = k;
    while jrev > 0 {
        jrev -= 1;
        let j = jrev;
        if *tau.add(j) == 0.0 { continue; }

        let mut sub_sq = 0.0f64;
        for i in (j + 1)..rows { let vi = *a.add(i * cols + j); sub_sq += vi * vi; }
        let vtv2 = 2.0 / *tau.add(j);
        let v0sq = vtv2 - sub_sq;
        let v0 = if v0sq > 0.0 { sqrt_f64_scalar(v0sq) } else { 0.0 };

        for col in 0..k {
            let mut dot = v0 * *q.add(j * k + col);
            for i in (j + 1)..rows { dot += *a.add(i * cols + j) * *q.add(i * k + col); }
            let factor = *tau.add(j) * dot;
            *q.add(j * k + col) -= factor * v0;
            for i in (j + 1)..rows { *q.add(i * k + col) -= factor * *a.add(i * cols + j); }
        }
    }
}

// ─── lstsq: solve Ax=b via QR ──────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn lstsq_f64(a: *mut f64, b: *const f64, x: *mut f64, scratch: *mut f64, m: u32, n: u32) {
    let rows = m as usize;
    let cols = n as usize;
    let k = if rows < cols { rows } else { cols };

    let a_copy = scratch;
    let q_ptr = a_copy.add(rows * cols);
    let r_ptr = q_ptr.add(rows * k);
    let tau_ptr = r_ptr.add(k * cols);
    let qtb_ptr = tau_ptr.add(k);
    let qr_scratch = qtb_ptr.add(k);

    copy_f64(a_copy, a as *const f64, rows * cols);
    qr_f64(a_copy, q_ptr, r_ptr, tau_ptr, qr_scratch, m, n);

    for i in 0..k {
        let mut sum = 0.0f64;
        for j in 0..rows { sum += *q_ptr.add(j * k + i) * *b.add(j); }
        *qtb_ptr.add(i) = sum;
    }

    let mut ii = k;
    while ii > 0 {
        ii -= 1;
        let mut sum = *qtb_ptr.add(ii);
        for j in (ii + 1)..cols { sum -= *r_ptr.add(ii * cols + j) * *x.add(j); }
        let diag = *r_ptr.add(ii * cols + ii);
        *x.add(ii) = if diag != 0.0 { sum / diag } else { 0.0 };
    }
}
