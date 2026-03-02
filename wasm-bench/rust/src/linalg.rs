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
