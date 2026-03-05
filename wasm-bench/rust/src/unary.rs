// Unary elementwise kernels for f32/f64
// SIMD-native: sqrt, abs, neg, ceil, floor (WASM opcodes)
// Libm: exp, log, sin, cos (no SIMD equivalent)

use core::arch::wasm32::*;
use crate::simd::{load_f64x2, store_f64x2, load_f32x4, store_f32x4};

// ─── SIMD macros — safe inner + thin unsafe FFI wrapper ─────────────────────

macro_rules! unary_simd_f64 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(inp: *const f64, out: *mut f64, n: u32) {

            // Safe inner function that operates on slices, so we can use safe indexing and iterators.
            fn inner(input: &[f64], output: &mut [f64]) {
                let len = input.len();
                let mut i = 0;
                while i + 4 <= len {
                    let v0 = load_f64x2(input, i);
                    let v1 = load_f64x2(input, i + 2);
                    store_f64x2(output, i, $op(v0));
                    store_f64x2(output, i + 2, $op(v1));
                    i += 4;
                }
                while i + 2 <= len {
                    let v = load_f64x2(input, i);
                    store_f64x2(output, i, $op(v));
                    i += 2;
                }
                while i < len {
                    let v = f64x2_splat(input[i]);
                    output[i] = f64x2_extract_lane::<0>($op(v));
                    i += 1;
                }
            }
                        
            // Call the inner function with slices; unsafe only for dereferencing & slice creation
            let len = n as usize;
            inner(
                core::slice::from_raw_parts(inp, len),
                core::slice::from_raw_parts_mut(out, len),
            );
        }
    };
}

macro_rules! unary_simd_f32 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(inp: *const f32, out: *mut f32, n: u32) {

            // Safe inner function that operates on slices, so we can use safe indexing and iterators.
            fn inner(input: &[f32], output: &mut [f32]) {
                let len = input.len();
                let mut i = 0;
                while i + 8 <= len {
                    let v0 = load_f32x4(input, i);
                    let v1 = load_f32x4(input, i + 4);
                    store_f32x4(output, i, $op(v0));
                    store_f32x4(output, i + 4, $op(v1));
                    i += 8;
                }
                while i + 4 <= len {
                    let v = load_f32x4(input, i);
                    store_f32x4(output, i, $op(v));
                    i += 4;
                }
                while i < len {
                    let v = f32x4_splat(input[i]);
                    output[i] = f32x4_extract_lane::<0>($op(v));
                    i += 1;
                }
            }
                        
            // Call the inner function with slices; unsafe only for dereferencing & slice creation
            let len = n as usize;
            inner(
                core::slice::from_raw_parts(inp, len),
                core::slice::from_raw_parts_mut(out, len),
            );
        }
    };
}

// ─── Libm scalar macros ─────────────────────────────────────────────────────

macro_rules! unary_libm_f64 {
    ($name:ident, $op:path) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(inp: *const f64, out: *mut f64, n: u32) {

            // Safe inner function that operates on slices, so we can use safe indexing and iterators.
            fn inner(input: &[f64], output: &mut [f64]) {
                for i in 0..input.len() { output[i] = $op(input[i]); }
            }
                        
            // Call the inner function with slices; unsafe only for dereferencing & slice creation
            let len = n as usize;
            inner(
                core::slice::from_raw_parts(inp, len),
                core::slice::from_raw_parts_mut(out, len),
            );
        }
    };
}

macro_rules! unary_libm_f32 {
    ($name:ident, $op:path) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(inp: *const f32, out: *mut f32, n: u32) {

            // Safe inner function that operates on slices, so we can use safe indexing and iterators.
            fn inner(input: &[f32], output: &mut [f32]) {
                for i in 0..input.len() { output[i] = $op(input[i]); }
            }
                        
            // Call the inner function with slices; unsafe only for dereferencing & slice creation
            let len = n as usize;
            inner(
                core::slice::from_raw_parts(inp, len),
                core::slice::from_raw_parts_mut(out, len),
            );
        }
    };
}

// ─── f64 SIMD ops ──────────────────────────────────────────────────────────

unary_simd_f64!(sqrt_f64, f64x2_sqrt);
unary_simd_f64!(abs_f64, f64x2_abs);
unary_simd_f64!(neg_f64, f64x2_neg);
unary_simd_f64!(ceil_f64, f64x2_ceil);
unary_simd_f64!(floor_f64, f64x2_floor);

// ─── f64 libm ops ──────────────────────────────────────────────────────────

unary_libm_f64!(exp_f64, libm::exp);
unary_libm_f64!(log_f64, libm::log);
unary_libm_f64!(sin_f64, libm::sin);
unary_libm_f64!(cos_f64, libm::cos);

// ─── f32 SIMD ops ──────────────────────────────────────────────────────────

unary_simd_f32!(sqrt_f32, f32x4_sqrt);
unary_simd_f32!(abs_f32, f32x4_abs);
unary_simd_f32!(neg_f32, f32x4_neg);
unary_simd_f32!(ceil_f32, f32x4_ceil);
unary_simd_f32!(floor_f32, f32x4_floor);

// ─── f32 libm ops ──────────────────────────────────────────────────────────

unary_libm_f32!(exp_f32, libm::expf);
unary_libm_f32!(log_f32, libm::logf);
unary_libm_f32!(sin_f32, libm::sinf);
unary_libm_f32!(cos_f32, libm::cosf);

// ─── sinh, cosh, tanh (libm scalar) ───────────────────────────────────

unary_libm_f64!(sinh_f64, libm::sinh);
unary_libm_f64!(cosh_f64, libm::cosh);
unary_libm_f64!(tanh_f64, libm::tanh);
unary_libm_f32!(sinh_f32, libm::sinhf);
unary_libm_f32!(cosh_f32, libm::coshf);
unary_libm_f32!(tanh_f32, libm::tanhf);

// ─── exp2: 2^x ─────────────────────────────────────────────────────────

unary_libm_f64!(exp2_f64, libm::exp2);
unary_libm_f32!(exp2_f32, libm::exp2f);

// ─── tan (sin/cos) ─────────────────────────────────────────────────────────

unary_libm_f64!(tan_f64, libm::tan);
unary_libm_f32!(tan_f32, libm::tanf);

// ─── signbit: 1.0 if sign bit set, 0.0 otherwise ───────────────────────────

fn signbit_f64_inner(input: &[f64], output: &mut [f64]) {
    let len = input.len();
    let sign_mask = i64x2_splat(0x8000000000000000u64 as i64);
    let one = f64x2_splat(1.0);
    let zero = f64x2_splat(0.0);
    let mut i = 0;
    while i + 2 <= len {
        let v = load_f64x2(input, i);
        let has_sign = v128_and(v, sign_mask);
        let mask = i64x2_ne(has_sign, i64x2_splat(0));
        store_f64x2(output, i, v128_bitselect(one, zero, mask));
        i += 2;
    }
    while i < len {
        output[i] = if input[i].to_bits() >> 63 != 0 { 1.0 } else { 0.0 };
        i += 1;
    }
}

fn signbit_f32_inner(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let sign_mask = i32x4_splat(0x80000000u32 as i32);
    let one = f32x4_splat(1.0);
    let zero = f32x4_splat(0.0);
    let mut i = 0;
    while i + 4 <= len {
        let v = load_f32x4(input, i);
        let has_sign = v128_and(v, sign_mask);
        let mask = i32x4_ne(has_sign, i32x4_splat(0));
        store_f32x4(output, i, v128_bitselect(one, zero, mask));
        i += 4;
    }
    while i < len {
        output[i] = if input[i].to_bits() >> 31 != 0 { 1.0 } else { 0.0 };
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn signbit_f64(inp: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    signbit_f64_inner(
        core::slice::from_raw_parts(inp, len),
        core::slice::from_raw_parts_mut(out, len),
    );
}

#[no_mangle]
pub unsafe extern "C" fn signbit_f32(inp: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    signbit_f32_inner(
        core::slice::from_raw_parts(inp, len),
        core::slice::from_raw_parts_mut(out, len),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX UNARY OPS (c128, c64)
// ═══════════════════════════════════════════════════════════════════════════

fn abs_c128_inner(input: &[f64], output: &mut [f64]) {
    let n = output.len();
    for i in 0..n {
        let re = input[2 * i];
        let im = input[2 * i + 1];
        output[i] = libm::sqrt(re * re + im * im);
    }
}

fn abs_c64_inner(input: &[f32], output: &mut [f32]) {
    let n = output.len();
    for i in 0..n {
        let re = input[2 * i];
        let im = input[2 * i + 1];
        output[i] = libm::sqrtf(re * re + im * im);
    }
}

fn exp_c128_inner(input: &[f64], output: &mut [f64]) {
    let n = output.len() / 2;
    for i in 0..n {
        let re = input[2 * i];
        let im = input[2 * i + 1];
        let ea = libm::exp(re);
        output[2 * i] = ea * libm::cos(im);
        output[2 * i + 1] = ea * libm::sin(im);
    }
}

fn exp_c64_inner(input: &[f32], output: &mut [f32]) {
    let n = output.len() / 2;
    for i in 0..n {
        let re = input[2 * i];
        let im = input[2 * i + 1];
        let ea = libm::expf(re);
        output[2 * i] = ea * libm::cosf(im);
        output[2 * i + 1] = ea * libm::sinf(im);
    }
}

#[no_mangle]
pub unsafe extern "C" fn abs_c128(inp: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    abs_c128_inner(
        core::slice::from_raw_parts(inp, len * 2),
        core::slice::from_raw_parts_mut(out, len),
    );
}

#[no_mangle]
pub unsafe extern "C" fn abs_c64(inp: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    abs_c64_inner(
        core::slice::from_raw_parts(inp, len * 2),
        core::slice::from_raw_parts_mut(out, len),
    );
}

#[no_mangle]
pub unsafe extern "C" fn exp_c128(inp: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    exp_c128_inner(
        core::slice::from_raw_parts(inp, len * 2),
        core::slice::from_raw_parts_mut(out, len * 2),
    );
}

#[no_mangle]
pub unsafe extern "C" fn exp_c64(inp: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    exp_c64_inner(
        core::slice::from_raw_parts(inp, len * 2),
        core::slice::from_raw_parts_mut(out, len * 2),
    );
}
