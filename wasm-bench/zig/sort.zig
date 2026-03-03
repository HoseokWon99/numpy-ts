// WASM sort kernels: quicksort, argsort, partition, argpartition for f32/f64
// Median-of-three pivot + insertion sort for small partitions

const INSERTION_THRESHOLD = 16;

// ─── Generic quicksort ──────────────────────────────────────────────────────

fn insertionSort(comptime T: type, data: [*]T, lo: usize, hi: usize) void {
    if (hi <= lo) return;
    var i: usize = lo + 1;
    while (i <= hi) : (i += 1) {
        const key = data[i];
        var j: usize = i;
        while (j > lo and data[j - 1] > key) : (j -= 1) {
            data[j] = data[j - 1];
        }
        data[j] = key;
    }
}

fn medianOfThree(comptime T: type, data: [*]T, lo: usize, hi: usize) usize {
    const mid = lo + (hi - lo) / 2;
    if (data[lo] > data[mid]) swap(T, data, lo, mid);
    if (data[lo] > data[hi]) swap(T, data, lo, hi);
    if (data[mid] > data[hi]) swap(T, data, mid, hi);
    return mid;
}

fn swap(comptime T: type, data: [*]T, a: usize, b: usize) void {
    const tmp = data[a];
    data[a] = data[b];
    data[b] = tmp;
}

fn partition(comptime T: type, data: [*]T, lo: usize, hi: usize) usize {
    const pivotIdx = medianOfThree(T, data, lo, hi);
    const pivot = data[pivotIdx];
    swap(T, data, pivotIdx, hi);
    var i: usize = lo;
    var j: usize = if (hi > 0) hi - 1 else 0;
    if (hi == 0) return lo;
    while (true) {
        while (i <= j and data[i] < pivot) : (i += 1) {}
        while (j > i and data[j] > pivot) : (j -= 1) {}
        if (i >= j) break;
        swap(T, data, i, j);
        i += 1;
        if (j > 0) j -= 1;
    }
    swap(T, data, i, hi);
    return i;
}

fn quicksort(comptime T: type, data: [*]T, lo: usize, hi: usize) void {
    if (hi <= lo) return;
    if (hi - lo + 1 <= INSERTION_THRESHOLD) {
        insertionSort(T, data, lo, hi);
        return;
    }
    const p = partition(T, data, lo, hi);
    if (p > 0) quicksort(T, data, lo, p -| 1);
    if (p < hi) quicksort(T, data, p + 1, hi);
}

// ─── Argsort: sort indices by comparing values ──────────────────────────────

fn insertionSortIdx(comptime T: type, vals: [*]const T, idx: [*]u32, lo: usize, hi: usize) void {
    if (hi <= lo) return;
    var i: usize = lo + 1;
    while (i <= hi) : (i += 1) {
        const key_idx = idx[i];
        const key_val = vals[key_idx];
        var j: usize = i;
        while (j > lo and vals[idx[j - 1]] > key_val) : (j -= 1) {
            idx[j] = idx[j - 1];
        }
        idx[j] = key_idx;
    }
}

fn medianOfThreeIdx(comptime T: type, vals: [*]const T, idx: [*]u32, lo: usize, hi: usize) usize {
    const mid = lo + (hi - lo) / 2;
    if (vals[idx[lo]] > vals[idx[mid]]) swapU32(idx, lo, mid);
    if (vals[idx[lo]] > vals[idx[hi]]) swapU32(idx, lo, hi);
    if (vals[idx[mid]] > vals[idx[hi]]) swapU32(idx, mid, hi);
    return mid;
}

fn swapU32(data: [*]u32, a: usize, b: usize) void {
    const tmp = data[a];
    data[a] = data[b];
    data[b] = tmp;
}

fn partitionIdx(comptime T: type, vals: [*]const T, idx: [*]u32, lo: usize, hi: usize) usize {
    const pivotPos = medianOfThreeIdx(T, vals, idx, lo, hi);
    const pivot = vals[idx[pivotPos]];
    swapU32(idx, pivotPos, hi);
    var i: usize = lo;
    var j: usize = if (hi > 0) hi - 1 else 0;
    if (hi == 0) return lo;
    while (true) {
        while (i <= j and vals[idx[i]] < pivot) : (i += 1) {}
        while (j > i and vals[idx[j]] > pivot) : (j -= 1) {}
        if (i >= j) break;
        swapU32(idx, i, j);
        i += 1;
        if (j > 0) j -= 1;
    }
    swapU32(idx, i, hi);
    return i;
}

fn quicksortIdx(comptime T: type, vals: [*]const T, idx: [*]u32, lo: usize, hi: usize) void {
    if (hi <= lo) return;
    if (hi - lo + 1 <= INSERTION_THRESHOLD) {
        insertionSortIdx(T, vals, idx, lo, hi);
        return;
    }
    const p = partitionIdx(T, vals, idx, lo, hi);
    if (p > 0) quicksortIdx(T, vals, idx, lo, p -| 1);
    if (p < hi) quicksortIdx(T, vals, idx, p + 1, hi);
}

// ─── Quickselect: partial sort so that arr[kth] is in sorted position ───────

fn quickselect(comptime T: type, data: [*]T, lo_in: usize, hi_in: usize, kth: usize) void {
    var lo = lo_in;
    var hi = hi_in;
    while (lo < hi) {
        if (hi - lo + 1 <= INSERTION_THRESHOLD) {
            insertionSort(T, data, lo, hi);
            return;
        }
        const p = partition(T, data, lo, hi);
        if (p == kth) return;
        if (kth < p) {
            hi = p -| 1;
        } else {
            lo = p + 1;
        }
    }
}

fn quickselectIdx(comptime T: type, vals: [*]const T, idx: [*]u32, lo_in: usize, hi_in: usize, kth: usize) void {
    var lo = lo_in;
    var hi = hi_in;
    while (lo < hi) {
        if (hi - lo + 1 <= INSERTION_THRESHOLD) {
            insertionSortIdx(T, vals, idx, lo, hi);
            return;
        }
        const p = partitionIdx(T, vals, idx, lo, hi);
        if (p == kth) return;
        if (kth < p) {
            hi = p -| 1;
        } else {
            lo = p + 1;
        }
    }
}

// ─── Exports ────────────────────────────────────────────────────────────────

export fn sort_f64(ptr: [*]f64, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    quicksort(f64, ptr, 0, len - 1);
}

export fn sort_f32(ptr: [*]f32, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    quicksort(f32, ptr, 0, len - 1);
}

export fn argsort_f64(vals: [*]const f64, idx: [*]u32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    quicksortIdx(f64, vals, idx, 0, len - 1);
}

export fn argsort_f32(vals: [*]const f32, idx: [*]u32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    quicksortIdx(f32, vals, idx, 0, len - 1);
}

export fn partition_f64(ptr: [*]f64, n: u32, kth: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    quickselect(f64, ptr, 0, len - 1, @as(usize, kth));
}

export fn partition_f32(ptr: [*]f32, n: u32, kth: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    quickselect(f32, ptr, 0, len - 1, @as(usize, kth));
}

export fn argpartition_f64(vals: [*]const f64, idx: [*]u32, n: u32, kth: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    quickselectIdx(f64, vals, idx, 0, len - 1, @as(usize, kth));
}

export fn argpartition_f32(vals: [*]const f32, idx: [*]u32, n: u32, kth: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    quickselectIdx(f32, vals, idx, 0, len - 1, @as(usize, kth));
}

// ─── Statistics: median, percentile, quantile ───────────────────────────
// All work in-place (destructive). Caller must copy data beforehand.

fn linearInterp(comptime T: type, data: [*]T, n: usize, frac: f64) T {
    // frac in [0,1] → virtual index
    const idx_f = frac * @as(f64, @floatFromInt(n - 1));
    const lo: usize = @intFromFloat(@floor(idx_f));
    const hi: usize = if (lo + 1 < n) lo + 1 else lo;
    const t: T = @floatCast(idx_f - @floor(idx_f));
    // Ensure both lo and hi are in sorted position
    quickselect(T, data, 0, n - 1, lo);
    if (hi != lo) quickselect(T, data, lo + 1, n - 1, hi);
    return data[lo] * (1 - t) + data[hi] * t;
}

export fn median_f64(ptr: [*]f64, n: u32) f64 {
    const len = @as(usize, n);
    if (len == 0) return 0;
    if (len == 1) return ptr[0];
    return linearInterp(f64, ptr, len, 0.5);
}
export fn median_f32(ptr: [*]f32, n: u32) f32 {
    const len = @as(usize, n);
    if (len == 0) return 0;
    if (len == 1) return ptr[0];
    return linearInterp(f32, ptr, len, 0.5);
}
export fn percentile_f64(ptr: [*]f64, n: u32, p: f64) f64 {
    const len = @as(usize, n);
    if (len == 0) return 0;
    if (len == 1) return ptr[0];
    return linearInterp(f64, ptr, len, p / 100.0);
}
export fn percentile_f32(ptr: [*]f32, n: u32, p: f64) f32 {
    const len = @as(usize, n);
    if (len == 0) return 0;
    if (len == 1) return ptr[0];
    return linearInterp(f32, ptr, len, p / 100.0);
}
export fn quantile_f64(ptr: [*]f64, n: u32, q: f64) f64 {
    const len = @as(usize, n);
    if (len == 0) return 0;
    if (len == 1) return ptr[0];
    return linearInterp(f64, ptr, len, q);
}
export fn quantile_f32(ptr: [*]f32, n: u32, q: f64) f32 {
    const len = @as(usize, n);
    if (len == 0) return 0;
    if (len == 1) return ptr[0];
    return linearInterp(f32, ptr, len, q);
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER SORT (i32, i16, i8) — trivial thanks to comptime T
// ═══════════════════════════════════════════════════════════════════════════

export fn sort_i32(ptr: [*]i32, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    quicksort(i32, ptr, 0, len - 1);
}
export fn sort_i16(ptr: [*]i16, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    quicksort(i16, ptr, 0, len - 1);
}
export fn sort_i8(ptr: [*]i8, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    quicksort(i8, ptr, 0, len - 1);
}

export fn argsort_i32(vals: [*]const i32, idx: [*]u32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    quicksortIdx(i32, vals, idx, 0, len - 1);
}
export fn argsort_i16(vals: [*]const i16, idx: [*]u32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    quicksortIdx(i16, vals, idx, 0, len - 1);
}
export fn argsort_i8(vals: [*]const i8, idx: [*]u32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    quicksortIdx(i8, vals, idx, 0, len - 1);
}
