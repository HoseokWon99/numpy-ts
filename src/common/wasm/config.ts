/**
 * Global WASM configuration.
 *
 * Controls whether WASM kernels are used and their size thresholds.
 * Useful for testing (force WASM path with multiplier=0, force JS with Infinity).
 */

/**
 * WASM memory configuration.
 *
 * Controls the size of the WASM linear memory pool used to back ArrayStorage
 * data directly (zero-copy). When the pool is full, allocations fall back to
 * regular JS TypedArrays (with copy-in/copy-out for WASM kernels).
 */
export const wasmMemoryConfig = {
  /** Total WASM linear memory size in bytes. Default 256 MiB. */
  maxMemoryBytes: 256 * 1024 * 1024,
  /** Scratch region for bump-allocating JS-fallback copy-ins. Default 4 MiB. */
  scratchBytes: 4 * 1024 * 1024,
  /** When true, fall back to JS-backed TypedArrays when WASM memory is full. */
  fallbackToJS: true,
};

export const wasmConfig = {
  /**
   * Multiplier applied to all WASM size thresholds.
   * - 1 (default): normal behavior, WASM used above optimal thresholds
   * - 0: always use WASM when structurally possible (for testing)
   * - Infinity: disable WASM entirely, always fall back to JS
   */
  thresholdMultiplier: 1,

  /**
   * Incremented each time a WASM kernel successfully executes.
   * Reset to 0 by callers (e.g. benchmark runner) to detect per-operation WASM usage.
   */
  wasmCallCount: 0,
};
