// Safe wrappers around WASM SIMD v128 load/store intrinsics.
// All computation code uses these instead of raw unsafe v128_load/v128_store.

use core::arch::wasm32::*;

// ─── f64x2 (2 × f64 = 16 bytes) ────────────────────────────────────────────

#[inline(always)]
pub fn load_f64x2(slice: &[f64], offset: usize) -> v128 {
    debug_assert!(offset + 2 <= slice.len());
    unsafe { v128_load(slice.as_ptr().add(offset) as *const v128) }
}

#[inline(always)]
pub fn store_f64x2(slice: &mut [f64], offset: usize, val: v128) {
    debug_assert!(offset + 2 <= slice.len());
    unsafe { v128_store(slice.as_mut_ptr().add(offset) as *mut v128, val) }
}

// ─── f32x4 (4 × f32 = 16 bytes) ────────────────────────────────────────────

#[inline(always)]
pub fn load_f32x4(slice: &[f32], offset: usize) -> v128 {
    debug_assert!(offset + 4 <= slice.len());
    unsafe { v128_load(slice.as_ptr().add(offset) as *const v128) }
}

#[inline(always)]
pub fn store_f32x4(slice: &mut [f32], offset: usize, val: v128) {
    debug_assert!(offset + 4 <= slice.len());
    unsafe { v128_store(slice.as_mut_ptr().add(offset) as *mut v128, val) }
}

// ─── i32x4 (4 × i32 = 16 bytes) ────────────────────────────────────────────

#[inline(always)]
pub fn load_i32x4(slice: &[i32], offset: usize) -> v128 {
    debug_assert!(offset + 4 <= slice.len());
    unsafe { v128_load(slice.as_ptr().add(offset) as *const v128) }
}

#[inline(always)]
pub fn store_i32x4(slice: &mut [i32], offset: usize, val: v128) {
    debug_assert!(offset + 4 <= slice.len());
    unsafe { v128_store(slice.as_mut_ptr().add(offset) as *mut v128, val) }
}

// ─── i16x8 (8 × i16 = 16 bytes) ────────────────────────────────────────────

#[inline(always)]
pub fn load_i16x8(slice: &[i16], offset: usize) -> v128 {
    debug_assert!(offset + 8 <= slice.len());
    unsafe { v128_load(slice.as_ptr().add(offset) as *const v128) }
}

#[inline(always)]
pub fn store_i16x8(slice: &mut [i16], offset: usize, val: v128) {
    debug_assert!(offset + 8 <= slice.len());
    unsafe { v128_store(slice.as_mut_ptr().add(offset) as *mut v128, val) }
}

// ─── i8x16 (16 × i8 = 16 bytes) ────────────────────────────────────────────

#[inline(always)]
pub fn load_i8x16(slice: &[i8], offset: usize) -> v128 {
    debug_assert!(offset + 16 <= slice.len());
    unsafe { v128_load(slice.as_ptr().add(offset) as *const v128) }
}

#[inline(always)]
pub fn store_i8x16(slice: &mut [i8], offset: usize, val: v128) {
    debug_assert!(offset + 16 <= slice.len());
    unsafe { v128_store(slice.as_mut_ptr().add(offset) as *mut v128, val) }
}

// ─── u32x4 (4 × u32 = 16 bytes) — for index arrays ────────────────────────

#[inline(always)]
pub fn load_u32x4(slice: &[u32], offset: usize) -> v128 {
    debug_assert!(offset + 4 <= slice.len());
    unsafe { v128_load(slice.as_ptr().add(offset) as *const v128) }
}

#[inline(always)]
pub fn store_u32x4(slice: &mut [u32], offset: usize, val: v128) {
    debug_assert!(offset + 4 <= slice.len());
    unsafe { v128_store(slice.as_mut_ptr().add(offset) as *mut v128, val) }
}
