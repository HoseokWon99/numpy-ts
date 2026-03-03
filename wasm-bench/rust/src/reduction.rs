// Reduction kernels for f32/f64
// Explicit WASM SIMD — LLVM can't autovectorize loop-carried dependencies
// without fast-math (which Rust doesn't expose)

use core::arch::wasm32::*;

#[no_mangle]
pub unsafe extern "C" fn sum_f64(ptr: *const f64, n: u32) -> f64 {
    let len = n as usize;
    let mut acc0 = f64x2_splat(0.0);
    let mut acc1 = f64x2_splat(0.0);
    let mut i = 0;
    while i + 4 <= len {
        acc0 = f64x2_add(acc0, v128_load(ptr.add(i) as *const v128));
        acc1 = f64x2_add(acc1, v128_load(ptr.add(i + 2) as *const v128));
        i += 4;
    }
    while i + 2 <= len {
        acc0 = f64x2_add(acc0, v128_load(ptr.add(i) as *const v128));
        i += 2;
    }
    acc0 = f64x2_add(acc0, acc1);
    let mut result = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
    while i < len { result += *ptr.add(i); i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn max_f64(ptr: *const f64, n: u32) -> f64 {
    let len = n as usize;
    if len == 0 { return f64::NEG_INFINITY; }
    let mut acc = f64x2_splat(*ptr);
    let mut i = 0;
    while i + 2 <= len {
        acc = f64x2_max(acc, v128_load(ptr.add(i) as *const v128));
        i += 2;
    }
    let a = f64x2_extract_lane::<0>(acc);
    let b = f64x2_extract_lane::<1>(acc);
    let mut result = if a > b { a } else { b };
    while i < len { let v = *ptr.add(i); if v > result { result = v; } i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn min_f64(ptr: *const f64, n: u32) -> f64 {
    let len = n as usize;
    if len == 0 { return f64::INFINITY; }
    let mut acc = f64x2_splat(*ptr);
    let mut i = 0;
    while i + 2 <= len {
        acc = f64x2_min(acc, v128_load(ptr.add(i) as *const v128));
        i += 2;
    }
    let a = f64x2_extract_lane::<0>(acc);
    let b = f64x2_extract_lane::<1>(acc);
    let mut result = if a < b { a } else { b };
    while i < len { let v = *ptr.add(i); if v < result { result = v; } i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn prod_f64(ptr: *const f64, n: u32) -> f64 {
    let len = n as usize;
    let mut acc0 = f64x2_splat(1.0);
    let mut acc1 = f64x2_splat(1.0);
    let mut i = 0;
    while i + 4 <= len {
        acc0 = f64x2_mul(acc0, v128_load(ptr.add(i) as *const v128));
        acc1 = f64x2_mul(acc1, v128_load(ptr.add(i + 2) as *const v128));
        i += 4;
    }
    while i + 2 <= len {
        acc0 = f64x2_mul(acc0, v128_load(ptr.add(i) as *const v128));
        i += 2;
    }
    acc0 = f64x2_mul(acc0, acc1);
    let mut result = f64x2_extract_lane::<0>(acc0) * f64x2_extract_lane::<1>(acc0);
    while i < len { result *= *ptr.add(i); i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn mean_f64(ptr: *const f64, n: u32) -> f64 {
    sum_f64(ptr, n) / n as f64
}

// ─── f32 ─────────────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn sum_f32(ptr: *const f32, n: u32) -> f32 {
    let len = n as usize;
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0;
    while i + 8 <= len {
        acc0 = f32x4_add(acc0, v128_load(ptr.add(i) as *const v128));
        acc1 = f32x4_add(acc1, v128_load(ptr.add(i + 4) as *const v128));
        i += 8;
    }
    while i + 4 <= len {
        acc0 = f32x4_add(acc0, v128_load(ptr.add(i) as *const v128));
        i += 4;
    }
    acc0 = f32x4_add(acc0, acc1);
    let mut result = f32x4_extract_lane::<0>(acc0)
        + f32x4_extract_lane::<1>(acc0)
        + f32x4_extract_lane::<2>(acc0)
        + f32x4_extract_lane::<3>(acc0);
    while i < len { result += *ptr.add(i); i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn max_f32(ptr: *const f32, n: u32) -> f32 {
    let len = n as usize;
    if len == 0 { return f32::NEG_INFINITY; }
    let mut acc = f32x4_splat(*ptr);
    let mut i = 0;
    while i + 4 <= len {
        acc = f32x4_max(acc, v128_load(ptr.add(i) as *const v128));
        i += 4;
    }
    let mut result = f32x4_extract_lane::<0>(acc);
    let v1 = f32x4_extract_lane::<1>(acc);
    let v2 = f32x4_extract_lane::<2>(acc);
    let v3 = f32x4_extract_lane::<3>(acc);
    if v1 > result { result = v1; }
    if v2 > result { result = v2; }
    if v3 > result { result = v3; }
    while i < len { let v = *ptr.add(i); if v > result { result = v; } i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn min_f32(ptr: *const f32, n: u32) -> f32 {
    let len = n as usize;
    if len == 0 { return f32::INFINITY; }
    let mut acc = f32x4_splat(*ptr);
    let mut i = 0;
    while i + 4 <= len {
        acc = f32x4_min(acc, v128_load(ptr.add(i) as *const v128));
        i += 4;
    }
    let mut result = f32x4_extract_lane::<0>(acc);
    let v1 = f32x4_extract_lane::<1>(acc);
    let v2 = f32x4_extract_lane::<2>(acc);
    let v3 = f32x4_extract_lane::<3>(acc);
    if v1 < result { result = v1; }
    if v2 < result { result = v2; }
    if v3 < result { result = v3; }
    while i < len { let v = *ptr.add(i); if v < result { result = v; } i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn prod_f32(ptr: *const f32, n: u32) -> f32 {
    let len = n as usize;
    let mut acc0 = f32x4_splat(1.0);
    let mut acc1 = f32x4_splat(1.0);
    let mut i = 0;
    while i + 8 <= len {
        acc0 = f32x4_mul(acc0, v128_load(ptr.add(i) as *const v128));
        acc1 = f32x4_mul(acc1, v128_load(ptr.add(i + 4) as *const v128));
        i += 8;
    }
    while i + 4 <= len {
        acc0 = f32x4_mul(acc0, v128_load(ptr.add(i) as *const v128));
        i += 4;
    }
    acc0 = f32x4_mul(acc0, acc1);
    let mut result = f32x4_extract_lane::<0>(acc0)
        * f32x4_extract_lane::<1>(acc0)
        * f32x4_extract_lane::<2>(acc0)
        * f32x4_extract_lane::<3>(acc0);
    while i < len { result *= *ptr.add(i); i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn mean_f32(ptr: *const f32, n: u32) -> f32 {
    sum_f32(ptr, n) / n as f32
}

// ─── nanmax: max ignoring NaN ───────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn nanmax_f64(ptr: *const f64, n: u32) -> f64 {
    let len = n as usize;
    if len == 0 { return f64::NEG_INFINITY; }
    let mut start = 0;
    while start < len && (*ptr.add(start)).is_nan() { start += 1; }
    if start == len { return *ptr; } // all NaN
    let mut result = *ptr.add(start);
    for i in (start + 1)..len {
        let v = *ptr.add(i);
        if !v.is_nan() && v > result { result = v; }
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn nanmax_f32(ptr: *const f32, n: u32) -> f32 {
    let len = n as usize;
    if len == 0 { return f32::NEG_INFINITY; }
    let mut start = 0;
    while start < len && (*ptr.add(start)).is_nan() { start += 1; }
    if start == len { return *ptr; }
    let mut result = *ptr.add(start);
    for i in (start + 1)..len {
        let v = *ptr.add(i);
        if !v.is_nan() && v > result { result = v; }
    }
    result
}

// ─── nanmin: min ignoring NaN ───────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn nanmin_f64(ptr: *const f64, n: u32) -> f64 {
    let len = n as usize;
    if len == 0 { return f64::INFINITY; }
    let mut start = 0;
    while start < len && (*ptr.add(start)).is_nan() { start += 1; }
    if start == len { return *ptr; }
    let mut result = *ptr.add(start);
    for i in (start + 1)..len {
        let v = *ptr.add(i);
        if !v.is_nan() && v < result { result = v; }
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn nanmin_f32(ptr: *const f32, n: u32) -> f32 {
    let len = n as usize;
    if len == 0 { return f32::INFINITY; }
    let mut start = 0;
    while start < len && (*ptr.add(start)).is_nan() { start += 1; }
    if start == len { return *ptr; }
    let mut result = *ptr.add(start);
    for i in (start + 1)..len {
        let v = *ptr.add(i);
        if !v.is_nan() && v < result { result = v; }
    }
    result
}

// ─── diff: first-order differences ──────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn diff_f64(inp: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    let out_len = len - 1;
    let mut i = 0;
    while i + 2 <= out_len {
        let v0 = v128_load(inp.add(i) as *const v128);
        let v1 = v128_load(inp.add(i + 1) as *const v128);
        v128_store(out.add(i) as *mut v128, f64x2_sub(v1, v0));
        i += 2;
    }
    while i < out_len {
        *out.add(i) = *inp.add(i + 1) - *inp.add(i);
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn diff_f32(inp: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    let out_len = len - 1;
    let mut i = 0;
    while i + 4 <= out_len {
        let v0 = v128_load(inp.add(i) as *const v128);
        let v1 = v128_load(inp.add(i + 1) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_sub(v1, v0));
        i += 4;
    }
    while i < out_len {
        *out.add(i) = *inp.add(i + 1) - *inp.add(i);
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER REDUCTIONS (i32, i16, i8)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn sum_i32(ptr: *const i32, n: u32) -> i32 {
    let len = n as usize;
    let mut acc0 = i32x4_splat(0);
    let mut acc1 = i32x4_splat(0);
    let mut i = 0;
    while i + 8 <= len {
        acc0 = i32x4_add(acc0, v128_load(ptr.add(i) as *const v128));
        acc1 = i32x4_add(acc1, v128_load(ptr.add(i + 4) as *const v128));
        i += 8;
    }
    while i + 4 <= len {
        acc0 = i32x4_add(acc0, v128_load(ptr.add(i) as *const v128));
        i += 4;
    }
    acc0 = i32x4_add(acc0, acc1);
    let mut result = i32x4_extract_lane::<0>(acc0)
        .wrapping_add(i32x4_extract_lane::<1>(acc0))
        .wrapping_add(i32x4_extract_lane::<2>(acc0))
        .wrapping_add(i32x4_extract_lane::<3>(acc0));
    while i < len { result = result.wrapping_add(*ptr.add(i)); i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn max_i32(ptr: *const i32, n: u32) -> i32 {
    let len = n as usize;
    if len == 0 { return i32::MIN; }
    let mut acc = i32x4_splat(*ptr);
    let mut i = 0;
    while i + 4 <= len {
        acc = i32x4_max(acc, v128_load(ptr.add(i) as *const v128));
        i += 4;
    }
    let mut result = i32x4_extract_lane::<0>(acc);
    let v1 = i32x4_extract_lane::<1>(acc);
    let v2 = i32x4_extract_lane::<2>(acc);
    let v3 = i32x4_extract_lane::<3>(acc);
    if v1 > result { result = v1; }
    if v2 > result { result = v2; }
    if v3 > result { result = v3; }
    while i < len { let v = *ptr.add(i); if v > result { result = v; } i += 1; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn min_i32(ptr: *const i32, n: u32) -> i32 {
    let len = n as usize;
    if len == 0 { return i32::MAX; }
    let mut acc = i32x4_splat(*ptr);
    let mut i = 0;
    while i + 4 <= len {
        acc = i32x4_min(acc, v128_load(ptr.add(i) as *const v128));
        i += 4;
    }
    let mut result = i32x4_extract_lane::<0>(acc);
    let v1 = i32x4_extract_lane::<1>(acc);
    let v2 = i32x4_extract_lane::<2>(acc);
    let v3 = i32x4_extract_lane::<3>(acc);
    if v1 < result { result = v1; }
    if v2 < result { result = v2; }
    if v3 < result { result = v3; }
    while i < len { let v = *ptr.add(i); if v < result { result = v; } i += 1; }
    result
}

// i16 reductions (widen to i32)
#[no_mangle]
pub unsafe extern "C" fn sum_i16(ptr: *const i16, n: u32) -> i32 {
    let len = n as usize;
    let mut result: i32 = 0;
    for i in 0..len { result = result.wrapping_add(*ptr.add(i) as i32); }
    result
}

#[no_mangle]
pub unsafe extern "C" fn max_i16(ptr: *const i16, n: u32) -> i16 {
    let len = n as usize;
    if len == 0 { return i16::MIN; }
    let mut result = *ptr;
    for i in 1..len { let v = *ptr.add(i); if v > result { result = v; } }
    result
}

#[no_mangle]
pub unsafe extern "C" fn min_i16(ptr: *const i16, n: u32) -> i16 {
    let len = n as usize;
    if len == 0 { return i16::MAX; }
    let mut result = *ptr;
    for i in 1..len { let v = *ptr.add(i); if v < result { result = v; } }
    result
}

// i8 reductions (widen to i32)
#[no_mangle]
pub unsafe extern "C" fn sum_i8(ptr: *const i8, n: u32) -> i32 {
    let len = n as usize;
    let mut result: i32 = 0;
    for i in 0..len { result = result.wrapping_add(*ptr.add(i) as i32); }
    result
}

#[no_mangle]
pub unsafe extern "C" fn max_i8(ptr: *const i8, n: u32) -> i8 {
    let len = n as usize;
    if len == 0 { return i8::MIN; }
    let mut result = *ptr;
    for i in 1..len { let v = *ptr.add(i); if v > result { result = v; } }
    result
}

#[no_mangle]
pub unsafe extern "C" fn min_i8(ptr: *const i8, n: u32) -> i8 {
    let len = n as usize;
    if len == 0 { return i8::MAX; }
    let mut result = *ptr;
    for i in 1..len { let v = *ptr.add(i); if v < result { result = v; } }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX REDUCTIONS (c128, c64)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn sum_c128(ptr: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    let mut acc0 = f64x2_splat(0.0);
    let mut acc1 = f64x2_splat(0.0);
    let mut i = 0;
    while i + 4 <= len {
        acc0 = f64x2_add(acc0, v128_load(ptr.add(i * 2) as *const v128));
        acc0 = f64x2_add(acc0, v128_load(ptr.add((i + 1) * 2) as *const v128));
        acc1 = f64x2_add(acc1, v128_load(ptr.add((i + 2) * 2) as *const v128));
        acc1 = f64x2_add(acc1, v128_load(ptr.add((i + 3) * 2) as *const v128));
        i += 4;
    }
    while i < len {
        acc0 = f64x2_add(acc0, v128_load(ptr.add(i * 2) as *const v128));
        i += 1;
    }
    acc0 = f64x2_add(acc0, acc1);
    *out = f64x2_extract_lane::<0>(acc0);
    *out.add(1) = f64x2_extract_lane::<1>(acc0);
}

#[no_mangle]
pub unsafe extern "C" fn sum_c64(ptr: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    let mut re_sum: f32 = 0.0;
    let mut im_sum: f32 = 0.0;
    for i in 0..len {
        re_sum += *ptr.add(2 * i);
        im_sum += *ptr.add(2 * i + 1);
    }
    *out = re_sum;
    *out.add(1) = im_sum;
}
