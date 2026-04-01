/**
 * WASM Memory Edge Cases: Allocator internals, scratch overflow, fromData
 *
 * Some tests use constrained memory (scratch overflow), others use default.
 * Since wasmMemoryConfig is set at module top level before lazy init,
 * all tests in this file share the same constrained config.
 */

import { wasmMemoryConfig, wasmConfig } from '../../src/common/wasm/config';

// Constrain memory BEFORE any WASM initialization
wasmMemoryConfig.maxMemoryBytes = 16 * 1024 * 1024; // 16 MiB total
wasmMemoryConfig.scratchBytes = 1 * 1024 * 1024; // 1 MiB scratch
// Heap: scratchBase(15 MiB) - heapBase(8 MiB) = 7 MiB usable

import { describe, it, expect, beforeEach } from 'vitest';
import {
  wasmMalloc,
  wasmFreeBytes,
  getSharedMemory,
  scratchAlloc,
  resetScratchAllocator,
} from '../../src/common/wasm/runtime';
import { heap_free } from '../../src/common/wasm/bins/alloc.wasm';
import { ArrayStorage } from '../../src/common/storage';
import { absolute as abs } from '../../src/full';
import { array } from '../../src';

beforeEach(() => {
  wasmConfig.thresholdMultiplier = 0; // force WASM
});

// ---------------------------------------------------------------------------
// Gap 6: wasmMalloc(0) returns null
// ---------------------------------------------------------------------------

describe('wasmMalloc edge cases', () => {
  it('wasmMalloc(0) returns null', () => {
    const region = wasmMalloc(0);
    expect(region).toBeNull();
  });

  it('wasmMalloc(-1) returns null', () => {
    const region = wasmMalloc(-1);
    expect(region).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Gap 6: heap_free(0) is safe
// ---------------------------------------------------------------------------

describe('heap_free edge cases', () => {
  it('heap_free(0) is a no-op', () => {
    const freeBefore = wasmFreeBytes();
    expect(() => heap_free(0)).not.toThrow();
    expect(wasmFreeBytes()).toBe(freeBefore);
  });
});

// ---------------------------------------------------------------------------
// Gap 6: Free-list recycling
// ---------------------------------------------------------------------------

describe('free-list recycling', () => {
  it('alloc → free → alloc reuses the freed block', () => {
    const freeAtStart = wasmFreeBytes();

    // First alloc
    const region1 = wasmMalloc(1024);
    expect(region1).not.toBeNull();
    const freeAfterAlloc1 = wasmFreeBytes();
    expect(freeAfterAlloc1).toBeLessThan(freeAtStart);

    const ptr1 = region1!.ptr;

    // Free
    region1!.release();
    const freeAfterRelease = wasmFreeBytes();
    expect(freeAfterRelease).toBeGreaterThan(freeAfterAlloc1);

    // Second alloc (same size — should reuse the freed block)
    const region2 = wasmMalloc(1024);
    expect(region2).not.toBeNull();
    expect(region2!.ptr).toBe(ptr1); // same pointer = recycled

    region2!.release();
  });

  it('free-list recycles across multiple alloc/free cycles', () => {
    const regions = [];
    for (let i = 0; i < 5; i++) {
      const r = wasmMalloc(2048);
      expect(r).not.toBeNull();
      regions.push(r!);
    }

    // Free all
    const ptrs = regions.map((r) => r.ptr);
    for (const r of regions) r.release();

    // Re-allocate — should get recycled pointers (in reverse free-list order)
    const newRegions = [];
    for (let i = 0; i < 5; i++) {
      const r = wasmMalloc(2048);
      expect(r).not.toBeNull();
      newRegions.push(r!);
    }

    // All new pointers should be from the recycled set
    const newPtrs = new Set(newRegions.map((r) => r.ptr));
    for (const ptr of ptrs) {
      expect(newPtrs.has(ptr)).toBe(true);
    }

    for (const r of newRegions) r.release();
  });
});

// ---------------------------------------------------------------------------
// Gap 6: Max size class boundary
// ---------------------------------------------------------------------------

describe('max size class boundary', () => {
  it('allocation above max size class (16 MiB) falls back to JS', () => {
    // Max size class is 2^24 = 16 MiB. Our heap is only 7 MiB anyway,
    // but even with a larger heap, 33 MiB exceeds all size classes.
    const huge = wasmMalloc(33 * 1024 * 1024);
    expect(huge).toBeNull();
  });

  it('ArrayStorage.zeros falls back to JS for oversized allocation', () => {
    // Request an array larger than the heap
    const s = ArrayStorage.zeros([2_000_000], 'float64'); // 16 MB > 7 MiB heap
    expect(s.isWasmBacked).toBe(false);
    expect(s.dtype).toBe('float64');
    expect(s.size).toBe(2_000_000);
    expect(s.data[0]).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Gap 5: Scratch region overflow
// ---------------------------------------------------------------------------

describe('scratch overflow', () => {
  it('scratchAlloc throws when scratch is exhausted', () => {
    resetScratchAllocator();
    // Scratch is 1 MiB. Request 2 MiB — should throw.
    expect(() => scratchAlloc(2 * 1024 * 1024)).toThrow(/scratch OOM/i);
  });

  it('scratch recovers after overflow (reset per kernel call)', () => {
    resetScratchAllocator();
    // Overflow
    expect(() => scratchAlloc(2 * 1024 * 1024)).toThrow();

    // Reset and try a normal allocation — should succeed
    resetScratchAllocator();
    const ptr = scratchAlloc(1024);
    expect(ptr).toBeGreaterThan(0);
  });

  it('kernel call succeeds on a normally-allocated array after scratch overflow+reset', () => {
    // Trigger a scratch overflow
    resetScratchAllocator();
    expect(() => scratchAlloc(2 * 1024 * 1024)).toThrow();

    // Scratch is now in a bad state, but the next kernel call resets it.
    // Verify a normal operation still works (resetScratchAllocator is called
    // at the start of every kernel invocation).
    const a = array([-1, -2, 3]);
    const result = abs(a);
    expect(result.tolist()).toEqual([1, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// Gap 7: fromData with data already in WASM buffer
// ---------------------------------------------------------------------------

describe('fromData with WASM-buffer data', () => {
  it('skips copy when data.buffer === wasm memory buffer', () => {
    const mem = getSharedMemory();

    // Allocate a region in WASM heap to get a valid offset
    const region = wasmMalloc(800); // 100 float64s
    expect(region).not.toBeNull();
    const ptr = region!.ptr;

    // Write some data
    const view = new Float64Array(mem.buffer, ptr, 100);
    for (let i = 0; i < 100; i++) view[i] = i * 1.5;

    // Create ArrayStorage from this view — should detect buffer === mem.buffer
    const s = ArrayStorage.fromData(view, [100], 'float64');

    // It should NOT be WASM-backed (no WasmRegion passed — it's a raw view)
    expect(s.isWasmBacked).toBe(false);

    // But data should be correct (no copy, same buffer)
    expect(s.data.buffer).toBe(mem.buffer);
    expect(s.data[0]).toBe(0);
    expect(s.data[1]).toBe(1.5);
    expect(s.data[99]).toBe(99 * 1.5);

    // Clean up the manually allocated region
    region!.release();
  });
});
