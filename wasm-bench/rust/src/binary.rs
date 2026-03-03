// Binary elementwise kernels for f32/f64
// Explicit WASM SIMD intrinsics

use core::arch::wasm32::*;

// ─── Macros for binary ops ──────────────────────────────────────────────────

macro_rules! binary_simd_f64 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
            let len = n as usize;
            let mut i = 0;
            while i + 4 <= len {
                let a0 = v128_load(a.add(i) as *const v128);
                let a1 = v128_load(a.add(i + 2) as *const v128);
                let b0 = v128_load(b.add(i) as *const v128);
                let b1 = v128_load(b.add(i + 2) as *const v128);
                v128_store(out.add(i) as *mut v128, $op(a0, b0));
                v128_store(out.add(i + 2) as *mut v128, $op(a1, b1));
                i += 4;
            }
            while i + 2 <= len {
                let a0 = v128_load(a.add(i) as *const v128);
                let b0 = v128_load(b.add(i) as *const v128);
                v128_store(out.add(i) as *mut v128, $op(a0, b0));
                i += 2;
            }
            while i < len {
                *out.add(i) = {
                    let av = f64x2_splat(*a.add(i));
                    let bv = f64x2_splat(*b.add(i));
                    f64x2_extract_lane::<0>($op(av, bv))
                };
                i += 1;
            }
        }
    };
}

macro_rules! binary_simd_f32 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
            let len = n as usize;
            let mut i = 0;
            while i + 8 <= len {
                let a0 = v128_load(a.add(i) as *const v128);
                let a1 = v128_load(a.add(i + 4) as *const v128);
                let b0 = v128_load(b.add(i) as *const v128);
                let b1 = v128_load(b.add(i + 4) as *const v128);
                v128_store(out.add(i) as *mut v128, $op(a0, b0));
                v128_store(out.add(i + 4) as *mut v128, $op(a1, b1));
                i += 8;
            }
            while i + 4 <= len {
                v128_store(out.add(i) as *mut v128, $op(
                    v128_load(a.add(i) as *const v128),
                    v128_load(b.add(i) as *const v128),
                ));
                i += 4;
            }
            while i < len {
                *out.add(i) = {
                    let av = f32x4_splat(*a.add(i));
                    let bv = f32x4_splat(*b.add(i));
                    f32x4_extract_lane::<0>($op(av, bv))
                };
                i += 1;
            }
        }
    };
}

// ─── Arithmetic ─────────────────────────────────────────────────────────────

binary_simd_f64!(add_f64, f64x2_add);
binary_simd_f64!(sub_f64, f64x2_sub);
binary_simd_f64!(mul_f64, f64x2_mul);
binary_simd_f64!(div_f64, f64x2_div);
binary_simd_f32!(add_f32, f32x4_add);
binary_simd_f32!(sub_f32, f32x4_sub);
binary_simd_f32!(mul_f32, f32x4_mul);
binary_simd_f32!(div_f32, f32x4_div);

// ─── maximum / minimum ─────────────────────────────────────────────────────

binary_simd_f64!(maximum_f64, f64x2_max);
binary_simd_f64!(minimum_f64, f64x2_min);
binary_simd_f32!(maximum_f32, f32x4_max);
binary_simd_f32!(minimum_f32, f32x4_min);

// ─── copysign: magnitude of a, sign of b ────────────────────────────────────

unsafe fn copysign_v128_f64(a: v128, b: v128) -> v128 {
    let abs_mask = i64x2_splat(0x7FFFFFFFFFFFFFFFu64 as i64);
    let sign_mask = i64x2_splat(0x8000000000000000u64 as i64);
    v128_or(v128_and(a, abs_mask), v128_and(b, sign_mask))
}
unsafe fn copysign_v128_f32(a: v128, b: v128) -> v128 {
    let abs_mask = i32x4_splat(0x7FFFFFFFu32 as i32);
    let sign_mask = i32x4_splat(0x80000000u32 as i32);
    v128_or(v128_and(a, abs_mask), v128_and(b, sign_mask))
}
binary_simd_f64!(copysign_f64, copysign_v128_f64);
binary_simd_f32!(copysign_f32, copysign_v128_f32);

// ─── fmax / fmin: NaN-aware max/min ─────────────────────────────────────────

unsafe fn fmax_v128_f64(a: v128, b: v128) -> v128 {
    let a_nan = f64x2_ne(a, a);
    let b_nan = f64x2_ne(b, b);
    let max_val = f64x2_max(a, b);
    v128_bitselect(b, v128_bitselect(a, max_val, b_nan), a_nan)
}
unsafe fn fmin_v128_f64(a: v128, b: v128) -> v128 {
    let a_nan = f64x2_ne(a, a);
    let b_nan = f64x2_ne(b, b);
    let min_val = f64x2_min(a, b);
    v128_bitselect(b, v128_bitselect(a, min_val, b_nan), a_nan)
}
unsafe fn fmax_v128_f32(a: v128, b: v128) -> v128 {
    let a_nan = f32x4_ne(a, a);
    let b_nan = f32x4_ne(b, b);
    let max_val = f32x4_max(a, b);
    v128_bitselect(b, v128_bitselect(a, max_val, b_nan), a_nan)
}
unsafe fn fmin_v128_f32(a: v128, b: v128) -> v128 {
    let a_nan = f32x4_ne(a, a);
    let b_nan = f32x4_ne(b, b);
    let min_val = f32x4_min(a, b);
    v128_bitselect(b, v128_bitselect(a, min_val, b_nan), a_nan)
}
binary_simd_f64!(fmax_f64, fmax_v128_f64);
binary_simd_f64!(fmin_f64, fmin_v128_f64);
binary_simd_f32!(fmax_f32, fmax_v128_f32);
binary_simd_f32!(fmin_f32, fmin_v128_f32);

// ─── mod (floored remainder): a - floor(a/b) * b ─────────────────────────

unsafe fn mod_v128_f64(a: v128, b: v128) -> v128 {
    f64x2_sub(a, f64x2_mul(f64x2_floor(f64x2_div(a, b)), b))
}
unsafe fn mod_v128_f32(a: v128, b: v128) -> v128 {
    f32x4_sub(a, f32x4_mul(f32x4_floor(f32x4_div(a, b)), b))
}
binary_simd_f64!(mod_f64, mod_v128_f64);
binary_simd_f32!(mod_f32, mod_v128_f32);

// ─── floor_divide: floor(a / b) ─────────────────────────────────────────

unsafe fn floor_divide_v128_f64(a: v128, b: v128) -> v128 { f64x2_floor(f64x2_div(a, b)) }
unsafe fn floor_divide_v128_f32(a: v128, b: v128) -> v128 { f32x4_floor(f32x4_div(a, b)) }
binary_simd_f64!(floor_divide_f64, floor_divide_v128_f64);
binary_simd_f32!(floor_divide_f32, floor_divide_v128_f32);

// ─── hypot: sqrt(a² + b²) ──────────────────────────────────────────────

unsafe fn hypot_v128_f64(a: v128, b: v128) -> v128 {
    f64x2_sqrt(f64x2_add(f64x2_mul(a, a), f64x2_mul(b, b)))
}
unsafe fn hypot_v128_f32(a: v128, b: v128) -> v128 {
    f32x4_sqrt(f32x4_add(f32x4_mul(a, a), f32x4_mul(b, b)))
}
binary_simd_f64!(hypot_f64, hypot_v128_f64);
binary_simd_f32!(hypot_f32, hypot_v128_f32);

// ─── logical_and / logical_xor ──────────────────────────────────────────────

unsafe fn logical_and_v128_f64(a: v128, b: v128) -> v128 {
    let zero = f64x2_splat(0.0);
    let one = f64x2_splat(1.0);
    let mask = v128_and(f64x2_ne(a, zero), f64x2_ne(b, zero));
    v128_bitselect(one, zero, mask)
}
unsafe fn logical_xor_v128_f64(a: v128, b: v128) -> v128 {
    let zero = f64x2_splat(0.0);
    let one = f64x2_splat(1.0);
    let mask = v128_xor(f64x2_ne(a, zero), f64x2_ne(b, zero));
    v128_bitselect(one, zero, mask)
}
unsafe fn logical_and_v128_f32(a: v128, b: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let one = f32x4_splat(1.0);
    v128_bitselect(one, zero, v128_and(f32x4_ne(a, zero), f32x4_ne(b, zero)))
}
unsafe fn logical_xor_v128_f32(a: v128, b: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let one = f32x4_splat(1.0);
    v128_bitselect(one, zero, v128_xor(f32x4_ne(a, zero), f32x4_ne(b, zero)))
}
binary_simd_f64!(logical_and_f64, logical_and_v128_f64);
binary_simd_f64!(logical_xor_f64, logical_xor_v128_f64);
binary_simd_f32!(logical_and_f32, logical_and_v128_f32);
binary_simd_f32!(logical_xor_f32, logical_xor_v128_f32);

// ─── power: a^b (scalar, uses libm) ────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn power_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = libm::pow(*a.add(i), *b.add(i));
    }
}
#[no_mangle]
pub unsafe extern "C" fn power_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = libm::powf(*a.add(i), *b.add(i));
    }
}

// ─── logaddexp: log(exp(a) + exp(b)) (scalar, uses libm) ───────────────────

#[no_mangle]
pub unsafe extern "C" fn logaddexp_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = libm::log(libm::exp(*a.add(i)) + libm::exp(*b.add(i)));
    }
}
#[no_mangle]
pub unsafe extern "C" fn logaddexp_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = libm::logf(libm::expf(*a.add(i)) + libm::expf(*b.add(i)));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER TYPES (i32, i16, i8)
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! binary_simd_i32 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const i32, b: *const i32, out: *mut i32, n: u32) {
            let len = n as usize;
            let mut i = 0;
            while i + 8 <= len {
                let a0 = v128_load(a.add(i) as *const v128);
                let a1 = v128_load(a.add(i + 4) as *const v128);
                let b0 = v128_load(b.add(i) as *const v128);
                let b1 = v128_load(b.add(i + 4) as *const v128);
                v128_store(out.add(i) as *mut v128, $op(a0, b0));
                v128_store(out.add(i + 4) as *mut v128, $op(a1, b1));
                i += 8;
            }
            while i + 4 <= len {
                v128_store(out.add(i) as *mut v128, $op(
                    v128_load(a.add(i) as *const v128),
                    v128_load(b.add(i) as *const v128),
                ));
                i += 4;
            }
            while i < len {
                let av = i32x4_splat(*a.add(i));
                let bv = i32x4_splat(*b.add(i));
                *out.add(i) = i32x4_extract_lane::<0>($op(av, bv));
                i += 1;
            }
        }
    };
}

macro_rules! binary_simd_i16 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const i16, b: *const i16, out: *mut i16, n: u32) {
            let len = n as usize;
            let mut i = 0;
            while i + 16 <= len {
                let a0 = v128_load(a.add(i) as *const v128);
                let a1 = v128_load(a.add(i + 8) as *const v128);
                let b0 = v128_load(b.add(i) as *const v128);
                let b1 = v128_load(b.add(i + 8) as *const v128);
                v128_store(out.add(i) as *mut v128, $op(a0, b0));
                v128_store(out.add(i + 8) as *mut v128, $op(a1, b1));
                i += 16;
            }
            while i + 8 <= len {
                v128_store(out.add(i) as *mut v128, $op(
                    v128_load(a.add(i) as *const v128),
                    v128_load(b.add(i) as *const v128),
                ));
                i += 8;
            }
            while i < len {
                let av = i16x8_splat(*a.add(i));
                let bv = i16x8_splat(*b.add(i));
                *out.add(i) = i16x8_extract_lane::<0>($op(av, bv));
                i += 1;
            }
        }
    };
}

macro_rules! binary_simd_i8 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const i8, b: *const i8, out: *mut i8, n: u32) {
            let len = n as usize;
            let mut i = 0;
            while i + 32 <= len {
                let a0 = v128_load(a.add(i) as *const v128);
                let a1 = v128_load(a.add(i + 16) as *const v128);
                let b0 = v128_load(b.add(i) as *const v128);
                let b1 = v128_load(b.add(i + 16) as *const v128);
                v128_store(out.add(i) as *mut v128, $op(a0, b0));
                v128_store(out.add(i + 16) as *mut v128, $op(a1, b1));
                i += 32;
            }
            while i + 16 <= len {
                v128_store(out.add(i) as *mut v128, $op(
                    v128_load(a.add(i) as *const v128),
                    v128_load(b.add(i) as *const v128),
                ));
                i += 16;
            }
            while i < len {
                let av = i8x16_splat(*a.add(i));
                let bv = i8x16_splat(*b.add(i));
                *out.add(i) = i8x16_extract_lane::<0>($op(av, bv));
                i += 1;
            }
        }
    };
}

// i32 ops
binary_simd_i32!(add_i32, i32x4_add);
binary_simd_i32!(sub_i32, i32x4_sub);
binary_simd_i32!(mul_i32, i32x4_mul);
binary_simd_i32!(maximum_i32, i32x4_max);
binary_simd_i32!(minimum_i32, i32x4_min);

// i16 ops
binary_simd_i16!(add_i16, i16x8_add);
binary_simd_i16!(sub_i16, i16x8_sub);
binary_simd_i16!(mul_i16, i16x8_mul);
binary_simd_i16!(maximum_i16, i16x8_max);
binary_simd_i16!(minimum_i16, i16x8_min);

// i8 ops (no i8x16_mul in WASM SIMD)
binary_simd_i8!(add_i8, i8x16_add);
binary_simd_i8!(sub_i8, i8x16_sub);
binary_simd_i8!(maximum_i8, i8x16_max);
binary_simd_i8!(minimum_i8, i8x16_min);

// mul_i8: scalar fallback
#[no_mangle]
pub unsafe extern "C" fn mul_i8(a: *const i8, b: *const i8, out: *mut i8, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = (*a.add(i)).wrapping_mul(*b.add(i));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX TYPES (c128, c64)
// ═══════════════════════════════════════════════════════════════════════════

// add_c128: component-wise f64 add on 2N elements
#[no_mangle]
pub unsafe extern "C" fn add_c128(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    add_f64(a, b, out, n * 2);
}

// add_c64: component-wise f32 add on 2N elements
#[no_mangle]
pub unsafe extern "C" fn add_c64(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    add_f32(a, b, out, n * 2);
}

// mul_c128: scalar complex multiply
#[no_mangle]
pub unsafe extern "C" fn mul_c128(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    for i in 0..n as usize {
        let ar = *a.add(2 * i);
        let ai = *a.add(2 * i + 1);
        let br = *b.add(2 * i);
        let bi = *b.add(2 * i + 1);
        *out.add(2 * i) = ar * br - ai * bi;
        *out.add(2 * i + 1) = ar * bi + ai * br;
    }
}

// mul_c64: complex multiply with SIMD
#[no_mangle]
pub unsafe extern "C" fn mul_c64(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    for i in 0..len {
        let ar = *a.add(2 * i);
        let ai = *a.add(2 * i + 1);
        let br = *b.add(2 * i);
        let bi = *b.add(2 * i + 1);
        *out.add(2 * i) = ar * br - ai * bi;
        *out.add(2 * i + 1) = ar * bi + ai * br;
    }
}
