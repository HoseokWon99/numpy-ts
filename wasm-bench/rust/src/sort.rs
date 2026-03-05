// Sort kernels: quicksort, argsort, partition, argpartition for f32/f64

const INSERTION_THRESHOLD: usize = 16;

// ─── Value-based sort helpers (safe, slice-based) ───────────────────────────

fn insertion_sort<T: PartialOrd + Copy>(data: &mut [T], lo: usize, hi: usize) {
    let mut i = lo + 1;
    while i <= hi {
        let key = data[i];
        let mut j = i;
        while j > lo && data[j - 1] > key {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
        i += 1;
    }
}

fn median_of_three<T: PartialOrd + Copy>(data: &mut [T], lo: usize, hi: usize) -> usize {
    let mid = lo + (hi - lo) / 2;
    if data[lo] > data[mid] { data.swap(lo, mid); }
    if data[lo] > data[hi] { data.swap(lo, hi); }
    if data[mid] > data[hi] { data.swap(mid, hi); }
    mid
}

fn partition_vals<T: PartialOrd + Copy>(data: &mut [T], lo: usize, hi: usize) -> usize {
    let pivot_idx = median_of_three(data, lo, hi);
    let pivot = data[pivot_idx];
    data.swap(pivot_idx, hi);
    let mut i = lo;
    let mut j = if hi > 0 { hi - 1 } else { return lo };
    loop {
        while i <= j && data[i] < pivot { i += 1; }
        while j > i && data[j] > pivot { j -= 1; }
        if i >= j { break; }
        data.swap(i, j);
        i += 1;
        if j > 0 { j -= 1; }
    }
    data.swap(i, hi);
    i
}

fn quicksort<T: PartialOrd + Copy>(data: &mut [T], lo: usize, hi: usize) {
    if hi <= lo { return; }
    if hi - lo + 1 <= INSERTION_THRESHOLD {
        insertion_sort(data, lo, hi);
        return;
    }
    let p = partition_vals(data, lo, hi);
    if p > 0 { quicksort(data, lo, p.saturating_sub(1)); }
    if p < hi { quicksort(data, p + 1, hi); }
}

// ─── Index-based sort helpers (safe, slice-based) ───────────────────────────

fn insertion_sort_idx<T: PartialOrd + Copy>(vals: &[T], idx: &mut [u32], lo: usize, hi: usize) {
    let mut i = lo + 1;
    while i <= hi {
        let key_idx = idx[i];
        let key_val = vals[key_idx as usize];
        let mut j = i;
        while j > lo && vals[idx[j - 1] as usize] > key_val {
            idx[j] = idx[j - 1];
            j -= 1;
        }
        idx[j] = key_idx;
        i += 1;
    }
}

fn median_of_three_idx<T: PartialOrd + Copy>(vals: &[T], idx: &mut [u32], lo: usize, hi: usize) -> usize {
    let mid = lo + (hi - lo) / 2;
    if vals[idx[lo] as usize] > vals[idx[mid] as usize] { idx.swap(lo, mid); }
    if vals[idx[lo] as usize] > vals[idx[hi] as usize] { idx.swap(lo, hi); }
    if vals[idx[mid] as usize] > vals[idx[hi] as usize] { idx.swap(mid, hi); }
    mid
}

fn partition_idx<T: PartialOrd + Copy>(vals: &[T], idx: &mut [u32], lo: usize, hi: usize) -> usize {
    let pivot_pos = median_of_three_idx(vals, idx, lo, hi);
    let pivot = vals[idx[pivot_pos] as usize];
    idx.swap(pivot_pos, hi);
    let mut i = lo;
    let mut j = if hi > 0 { hi - 1 } else { return lo };
    loop {
        while i <= j && vals[idx[i] as usize] < pivot { i += 1; }
        while j > i && vals[idx[j] as usize] > pivot { j -= 1; }
        if i >= j { break; }
        idx.swap(i, j);
        i += 1;
        if j > 0 { j -= 1; }
    }
    idx.swap(i, hi);
    i
}

fn quicksort_idx<T: PartialOrd + Copy>(vals: &[T], idx: &mut [u32], lo: usize, hi: usize) {
    if hi <= lo { return; }
    if hi - lo + 1 <= INSERTION_THRESHOLD {
        insertion_sort_idx(vals, idx, lo, hi);
        return;
    }
    let p = partition_idx(vals, idx, lo, hi);
    if p > 0 { quicksort_idx(vals, idx, lo, p.saturating_sub(1)); }
    if p < hi { quicksort_idx(vals, idx, p + 1, hi); }
}

fn quickselect<T: PartialOrd + Copy>(data: &mut [T], mut lo: usize, mut hi: usize, kth: usize) {
    while lo < hi {
        if hi - lo + 1 <= INSERTION_THRESHOLD {
            insertion_sort(data, lo, hi);
            return;
        }
        let p = partition_vals(data, lo, hi);
        if p == kth { return; }
        if kth < p { hi = p.saturating_sub(1); } else { lo = p + 1; }
    }
}

fn quickselect_idx<T: PartialOrd + Copy>(vals: &[T], idx: &mut [u32], mut lo: usize, mut hi: usize, kth: usize) {
    while lo < hi {
        if hi - lo + 1 <= INSERTION_THRESHOLD {
            insertion_sort_idx(vals, idx, lo, hi);
            return;
        }
        let p = partition_idx(vals, idx, lo, hi);
        if p == kth { return; }
        if kth < p { hi = p.saturating_sub(1); } else { lo = p + 1; }
    }
}

// ─── Statistics helpers ─────────────────────────────────────────────────

fn linear_interp_f64(data: &mut [f64], frac: f64) -> f64 {
    let n = data.len();
    let idx_f = frac * (n - 1) as f64;
    let lo = libm::floor(idx_f) as usize;
    let hi = if lo + 1 < n { lo + 1 } else { lo };
    let t = idx_f - libm::floor(idx_f);
    quickselect(data, 0, n - 1, lo);
    if hi != lo { quickselect(data, lo + 1, n - 1, hi); }
    data[lo] * (1.0 - t) + data[hi] * t
}

fn linear_interp_f32(data: &mut [f32], frac: f64) -> f32 {
    let n = data.len();
    let idx_f = frac * (n - 1) as f64;
    let lo = libm::floor(idx_f) as usize;
    let hi = if lo + 1 < n { lo + 1 } else { lo };
    let t = (idx_f - libm::floor(idx_f)) as f32;
    quickselect(data, 0, n - 1, lo);
    if hi != lo { quickselect(data, lo + 1, n - 1, hi); }
    data[lo] * (1.0 - t) + data[hi] * t
}

// ─── FFI Exports ─────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn sort_f64(ptr: *mut f64, n: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quicksort(data, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn sort_f32(ptr: *mut f32, n: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quicksort(data, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn argsort_f64(vals: *const f64, idx: *mut u32, n: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len { ix[i] = i as u32; }
    if len <= 1 { return; }
    quicksort_idx(v, ix, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn argsort_f32(vals: *const f32, idx: *mut u32, n: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len { ix[i] = i as u32; }
    if len <= 1 { return; }
    quicksort_idx(v, ix, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn partition_f64(ptr: *mut f64, n: u32, kth: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quickselect(data, 0, len - 1, kth as usize);
}

#[no_mangle]
pub unsafe extern "C" fn partition_f32(ptr: *mut f32, n: u32, kth: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quickselect(data, 0, len - 1, kth as usize);
}

#[no_mangle]
pub unsafe extern "C" fn argpartition_f64(vals: *const f64, idx: *mut u32, n: u32, kth: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len { ix[i] = i as u32; }
    if len <= 1 { return; }
    quickselect_idx(v, ix, 0, len - 1, kth as usize);
}

#[no_mangle]
pub unsafe extern "C" fn argpartition_f32(vals: *const f32, idx: *mut u32, n: u32, kth: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len { ix[i] = i as u32; }
    if len <= 1 { return; }
    quickselect_idx(v, ix, 0, len - 1, kth as usize);
}

// ─── Statistics: median, percentile, quantile ───────────────────────────

#[no_mangle]
pub unsafe extern "C" fn median_f64(ptr: *mut f64, n: u32) -> f64 {
    let len = n as usize;
    if len == 0 { return 0.0; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 { return data[0]; }
    linear_interp_f64(data, 0.5)
}
#[no_mangle]
pub unsafe extern "C" fn median_f32(ptr: *mut f32, n: u32) -> f32 {
    let len = n as usize;
    if len == 0 { return 0.0; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 { return data[0]; }
    linear_interp_f32(data, 0.5)
}
#[no_mangle]
pub unsafe extern "C" fn percentile_f64(ptr: *mut f64, n: u32, p: f64) -> f64 {
    let len = n as usize;
    if len == 0 { return 0.0; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 { return data[0]; }
    linear_interp_f64(data, p / 100.0)
}
#[no_mangle]
pub unsafe extern "C" fn percentile_f32(ptr: *mut f32, n: u32, p: f64) -> f32 {
    let len = n as usize;
    if len == 0 { return 0.0; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 { return data[0]; }
    linear_interp_f32(data, p / 100.0)
}
#[no_mangle]
pub unsafe extern "C" fn quantile_f64(ptr: *mut f64, n: u32, q: f64) -> f64 {
    let len = n as usize;
    if len == 0 { return 0.0; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 { return data[0]; }
    linear_interp_f64(data, q)
}
#[no_mangle]
pub unsafe extern "C" fn quantile_f32(ptr: *mut f32, n: u32, q: f64) -> f32 {
    let len = n as usize;
    if len == 0 { return 0.0; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 { return data[0]; }
    linear_interp_f32(data, q)
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER SORT (i32, i16, i8) — generic quicksort<T>
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn sort_i32(ptr: *mut i32, n: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quicksort(data, 0, len - 1);
}
#[no_mangle]
pub unsafe extern "C" fn sort_i16(ptr: *mut i16, n: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quicksort(data, 0, len - 1);
}
#[no_mangle]
pub unsafe extern "C" fn sort_i8(ptr: *mut i8, n: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quicksort(data, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn argsort_i32(vals: *const i32, idx: *mut u32, n: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len { ix[i] = i as u32; }
    if len <= 1 { return; }
    quicksort_idx(v, ix, 0, len - 1);
}
#[no_mangle]
pub unsafe extern "C" fn argsort_i16(vals: *const i16, idx: *mut u32, n: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len { ix[i] = i as u32; }
    if len <= 1 { return; }
    quicksort_idx(v, ix, 0, len - 1);
}
#[no_mangle]
pub unsafe extern "C" fn argsort_i8(vals: *const i8, idx: *mut u32, n: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len { ix[i] = i as u32; }
    if len <= 1 { return; }
    quicksort_idx(v, ix, 0, len - 1);
}
