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
// INTEGER SORT — radix/counting sort for O(n) performance
// ═══════════════════════════════════════════════════════════════════════════

// ─── Counting sort for i8 (256 buckets, stack-only) ─────────────────────

fn countingSortI8(data: [*]i8, len: usize) void {
    var counts: [256]u32 = [_]u32{0} ** 256;
    for (0..len) |i| {
        const key = @as(u8, @bitCast(data[i])) +% 128; // signed → unsigned offset
        counts[key] += 1;
    }
    var pos: usize = 0;
    for (0..256) |bucket| {
        var c = counts[bucket];
        while (c > 0) : (c -= 1) {
            data[pos] = @bitCast(@as(u8, @intCast(bucket)) -% 128);
            pos += 1;
        }
    }
}

fn countingArgsortI8(vals: [*]const i8, idx: [*]u32, len: usize) void {
    // Count occurrences
    var counts: [256]u32 = [_]u32{0} ** 256;
    for (0..len) |i| {
        const key = @as(u8, @bitCast(vals[i])) +% 128;
        counts[key] += 1;
    }
    // Prefix sum → starting positions
    var offsets: [256]u32 = [_]u32{0} ** 256;
    var total: u32 = 0;
    for (0..256) |bucket| {
        offsets[bucket] = total;
        total += counts[bucket];
    }
    // Place indices in sorted order
    for (0..len) |i| {
        const key = @as(u8, @bitCast(vals[i])) +% 128;
        idx[offsets[key]] = @as(u32, @intCast(i));
        offsets[key] += 1;
    }
}

// ─── Counting sort for i16 (65536 buckets, stack-only ~256KB) ───────────

fn countingSortI16(data: [*]i16, len: usize) void {
    var counts: [65536]u32 = [_]u32{0} ** 65536;
    for (0..len) |i| {
        const key = @as(u16, @bitCast(data[i])) +% 32768;
        counts[key] += 1;
    }
    var pos: usize = 0;
    for (0..65536) |bucket| {
        var c = counts[bucket];
        while (c > 0) : (c -= 1) {
            data[pos] = @bitCast(@as(u16, @intCast(bucket)) -% 32768);
            pos += 1;
        }
    }
}

fn countingArgsortI16(vals: [*]const i16, idx: [*]u32, len: usize) void {
    var counts: [65536]u32 = [_]u32{0} ** 65536;
    for (0..len) |i| {
        const key = @as(u16, @bitCast(vals[i])) +% 32768;
        counts[key] += 1;
    }
    var offsets: [65536]u32 = [_]u32{0} ** 65536;
    var total: u32 = 0;
    for (0..65536) |bucket| {
        offsets[bucket] = total;
        total += counts[bucket];
    }
    for (0..len) |i| {
        const key = @as(u16, @bitCast(vals[i])) +% 32768;
        idx[offsets[key]] = @as(u32, @intCast(i));
        offsets[key] += 1;
    }
}

// ─── LSD Radix sort for i32 (4 passes × 256 buckets) ───────────────────
// Needs a scratch buffer — allocated via WASM memory.grow

fn wasmAllocScratch(bytes: usize) ?[*]u8 {
    const pages_needed = (bytes + 65535) / 65536; // round up to page boundary
    const old_pages = @wasmMemoryGrow(0, pages_needed);
    if (old_pages < 0) return null;
    return @ptrFromInt(@as(usize, @intCast(old_pages)) * 65536);
}

fn radixSortI32(data: [*]i32, len: usize) void {
    // Allocate scratch buffer
    const scratch_ptr = wasmAllocScratch(len * 4) orelse {
        // Fallback to quicksort if memory.grow fails
        quicksort(i32, data, 0, len - 1);
        return;
    };
    const tmp: [*]i32 = @ptrCast(@alignCast(scratch_ptr));

    // Flip sign bit so signed sort becomes unsigned sort
    for (0..len) |i| {
        data[i] = @bitCast(@as(u32, @bitCast(data[i])) ^ 0x80000000);
    }

    // 4-pass LSD radix sort (byte 0, 1, 2, 3)
    var src = data;
    var dst = tmp;
    for (0..4) |pass| {
        const shift: u5 = @intCast(pass * 8);
        var counts: [256]u32 = [_]u32{0} ** 256;

        // Count
        for (0..len) |i| {
            const key = (@as(u32, @bitCast(src[i])) >> shift) & 0xFF;
            counts[key] += 1;
        }

        // Prefix sum
        var offsets: [256]u32 = [_]u32{0} ** 256;
        var total: u32 = 0;
        for (0..256) |bucket| {
            offsets[bucket] = total;
            total += counts[bucket];
        }

        // Scatter
        for (0..len) |i| {
            const key = (@as(u32, @bitCast(src[i])) >> shift) & 0xFF;
            dst[offsets[key]] = src[i];
            offsets[key] += 1;
        }

        // Swap src/dst
        const t = src;
        src = dst;
        dst = t;
    }

    // After 4 passes (even), src == data, dst == tmp → result is in data
    // Flip sign bit back
    for (0..len) |i| {
        data[i] = @bitCast(@as(u32, @bitCast(data[i])) ^ 0x80000000);
    }
}

fn radixArgsortI32(vals: [*]const i32, idx: [*]u32, len: usize) void {
    // Allocate scratch buffer for temp indices
    const scratch_ptr = wasmAllocScratch(len * 4) orelse {
        quicksortIdx(i32, vals, idx, 0, len - 1);
        return;
    };
    const tmp: [*]u32 = @ptrCast(@alignCast(scratch_ptr));

    // 4-pass LSD radix sort on indices, keyed by (vals[idx[i]] ^ sign_flip)
    var src = idx;
    var dst = tmp;
    for (0..4) |pass| {
        const shift: u5 = @intCast(pass * 8);
        var counts: [256]u32 = [_]u32{0} ** 256;

        for (0..len) |i| {
            const v = @as(u32, @bitCast(vals[src[i]])) ^ 0x80000000;
            const key = (v >> shift) & 0xFF;
            counts[key] += 1;
        }

        var offsets: [256]u32 = [_]u32{0} ** 256;
        var total: u32 = 0;
        for (0..256) |bucket| {
            offsets[bucket] = total;
            total += counts[bucket];
        }

        for (0..len) |i| {
            const v = @as(u32, @bitCast(vals[src[i]])) ^ 0x80000000;
            const key = (v >> shift) & 0xFF;
            dst[offsets[key]] = src[i];
            offsets[key] += 1;
        }

        const t = src;
        src = dst;
        dst = t;
    }

    // After 4 passes (even), src == idx → result is in idx. Done.
}

// ─── Integer sort exports ───────────────────────────────────────────────

export fn sort_i32(ptr: [*]i32, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    radixSortI32(ptr, len);
}
export fn sort_i16(ptr: [*]i16, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    countingSortI16(ptr, len);
}
export fn sort_i8(ptr: [*]i8, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    countingSortI8(ptr, len);
}

export fn argsort_i32(vals: [*]const i32, idx: [*]u32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    radixArgsortI32(vals, idx, len);
}
export fn argsort_i16(vals: [*]const i16, idx: [*]u32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    countingArgsortI16(vals, idx, len);
}
export fn argsort_i8(vals: [*]const i8, idx: [*]u32, n: u32) void {
    const len = @as(usize, n);
    for (0..len) |i| idx[i] = @as(u32, @intCast(i));
    if (len <= 1) return;
    countingArgsortI8(vals, idx, len);
}
