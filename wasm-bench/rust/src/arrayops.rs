// Array operation kernels: roll, flip, tile, pad, take, gradient

use core::arch::wasm32::*;

// ─── SIMD copy/zero helpers ─────────────────────────────────────────────────

unsafe fn simd_copy_f64(dst: *mut f64, src: *const f64, n: usize) {
    let mut i = 0;
    while i + 4 <= n {
        v128_store(dst.add(i) as *mut v128, v128_load(src.add(i) as *const v128));
        v128_store(dst.add(i + 2) as *mut v128, v128_load(src.add(i + 2) as *const v128));
        i += 4;
    }
    while i + 2 <= n {
        v128_store(dst.add(i) as *mut v128, v128_load(src.add(i) as *const v128));
        i += 2;
    }
    while i < n { *dst.add(i) = *src.add(i); i += 1; }
}

unsafe fn simd_copy_f32(dst: *mut f32, src: *const f32, n: usize) {
    let mut i = 0;
    while i + 8 <= n {
        v128_store(dst.add(i) as *mut v128, v128_load(src.add(i) as *const v128));
        v128_store(dst.add(i + 4) as *mut v128, v128_load(src.add(i + 4) as *const v128));
        i += 8;
    }
    while i + 4 <= n {
        v128_store(dst.add(i) as *mut v128, v128_load(src.add(i) as *const v128));
        i += 4;
    }
    while i < n { *dst.add(i) = *src.add(i); i += 1; }
}

unsafe fn simd_zero_f64(dst: *mut f64, n: usize) {
    let zero = f64x2_splat(0.0);
    let mut i = 0;
    while i + 2 <= n { v128_store(dst.add(i) as *mut v128, zero); i += 2; }
    while i < n { *dst.add(i) = 0.0; i += 1; }
}

unsafe fn simd_zero_f32(dst: *mut f32, n: usize) {
    let zero = f32x4_splat(0.0);
    let mut i = 0;
    while i + 4 <= n { v128_store(dst.add(i) as *mut v128, zero); i += 4; }
    while i < n { *dst.add(i) = 0.0; i += 1; }
}

// ─── roll: circular shift ───────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn roll_f64(inp: *const f64, out: *mut f64, n: u32, shift: i32) {
    let len = n as usize;
    if len == 0 { return; }
    let s = ((shift as i64).rem_euclid(len as i64)) as usize;
    if s == 0 { simd_copy_f64(out, inp, len); return; }
    simd_copy_f64(out, inp.add(len - s), s);
    simd_copy_f64(out.add(s), inp, len - s);
}

#[no_mangle]
pub unsafe extern "C" fn roll_f32(inp: *const f32, out: *mut f32, n: u32, shift: i32) {
    let len = n as usize;
    if len == 0 { return; }
    let s = ((shift as i64).rem_euclid(len as i64)) as usize;
    if s == 0 { simd_copy_f32(out, inp, len); return; }
    simd_copy_f32(out, inp.add(len - s), s);
    simd_copy_f32(out.add(s), inp, len - s);
}

// ─── flip: reverse array ────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn flip_f64(inp: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    for i in 0..len { *out.add(i) = *inp.add(len - 1 - i); }
}

#[no_mangle]
pub unsafe extern "C" fn flip_f32(inp: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    for i in 0..len { *out.add(i) = *inp.add(len - 1 - i); }
}

// ─── tile: repeat array ─────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn tile_f64(inp: *const f64, out: *mut f64, n: u32, reps: u32) {
    let len = n as usize;
    for rep in 0..reps as usize {
        simd_copy_f64(out.add(rep * len), inp, len);
    }
}

#[no_mangle]
pub unsafe extern "C" fn tile_f32(inp: *const f32, out: *mut f32, n: u32, reps: u32) {
    let len = n as usize;
    for rep in 0..reps as usize {
        simd_copy_f32(out.add(rep * len), inp, len);
    }
}

// ─── pad: zero-pad 2D array ─────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn pad_f64(
    inp: *const f64, out: *mut f64, rows: u32, cols: u32, pw: u32,
) {
    let (r, c, p) = (rows as usize, cols as usize, pw as usize);
    let out_cols = c + 2 * p;
    simd_zero_f64(out, (r + 2 * p) * out_cols);
    for i in 0..r {
        simd_copy_f64(out.add((i + p) * out_cols + p), inp.add(i * c), c);
    }
}

#[no_mangle]
pub unsafe extern "C" fn pad_f32(
    inp: *const f32, out: *mut f32, rows: u32, cols: u32, pw: u32,
) {
    let (r, c, p) = (rows as usize, cols as usize, pw as usize);
    let out_cols = c + 2 * p;
    simd_zero_f32(out, (r + 2 * p) * out_cols);
    for i in 0..r {
        simd_copy_f32(out.add((i + p) * out_cols + p), inp.add(i * c), c);
    }
}

// ─── take: gather by index ──────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn take_f64(data: *const f64, indices: *const u32, out: *mut f64, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = *data.add(*indices.add(i) as usize);
    }
}

#[no_mangle]
pub unsafe extern "C" fn take_f32(data: *const f32, indices: *const u32, out: *mut f32, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = *data.add(*indices.add(i) as usize);
    }
}

// ─── gradient: numerical gradient (central differences) ─────────────────────

#[no_mangle]
pub unsafe extern "C" fn gradient_f64(inp: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    if len < 2 { return; }
    *out = *inp.add(1) - *inp;
    *out.add(len - 1) = *inp.add(len - 1) - *inp.add(len - 2);
    if len <= 2 { return; }
    let half = f64x2_splat(0.5);
    let mut i = 1;
    while i + 2 < len {
        let fwd = v128_load(inp.add(i + 1) as *const v128);
        let bwd = v128_load(inp.add(i - 1) as *const v128);
        v128_store(out.add(i) as *mut v128, f64x2_mul(f64x2_sub(fwd, bwd), half));
        i += 2;
    }
    while i < len - 1 {
        *out.add(i) = (*inp.add(i + 1) - *inp.add(i - 1)) * 0.5;
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn gradient_f32(inp: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    if len < 2 { return; }
    *out = *inp.add(1) - *inp;
    *out.add(len - 1) = *inp.add(len - 1) - *inp.add(len - 2);
    if len <= 2 { return; }
    let half = f32x4_splat(0.5);
    let mut i = 1;
    while i + 4 < len {
        let fwd = v128_load(inp.add(i + 1) as *const v128);
        let bwd = v128_load(inp.add(i - 1) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_mul(f32x4_sub(fwd, bwd), half));
        i += 4;
    }
    while i < len - 1 {
        *out.add(i) = (*inp.add(i + 1) - *inp.add(i - 1)) * 0.5;
        i += 1;
    }
}

// ─── nonzero: return indices of non-zero elements ────────────────────────

#[no_mangle]
pub unsafe extern "C" fn nonzero_f64(ptr: *const f64, out: *mut u32, n: u32) -> u32 {
    let len = n as usize;
    let mut count: usize = 0;
    for i in 0..len {
        if *ptr.add(i) != 0.0 {
            *out.add(count) = i as u32;
            count += 1;
        }
    }
    count as u32
}

#[no_mangle]
pub unsafe extern "C" fn nonzero_f32(ptr: *const f32, out: *mut u32, n: u32) -> u32 {
    let len = n as usize;
    let mut count: usize = 0;
    for i in 0..len {
        if *ptr.add(i) != 0.0 {
            *out.add(count) = i as u32;
            count += 1;
        }
    }
    count as u32
}
